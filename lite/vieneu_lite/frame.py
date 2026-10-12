"""One CUDA graph per audio frame over ``B`` fixed slots (design §3).

A replay runs, for every slot at once:

    acoustic decoder (EOS + 16 codes, per-row penalty/sampling)
    -> bookkeeping (codes ring, finished, length)
    -> backbone step over a static ring KV cache -> next hidden

Rows enter a slot with ``admit`` (host side, between replays) and are never
removed: an empty slot keeps stepping on stale state and its output is ignored,
so shapes never change and the graph is captured once.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class RepHistory:
    """Sliding-window repetition history on the device, ``B`` rows x ``n_vq`` codebooks.

    ``counts[b, ch, code]`` counts ``code`` in row ``b``'s last ``window`` frames;
    ``ring[b, ch, f % window]`` remembers the code of frame ``f`` so it can be
    un-counted when it leaves the window; ``start[b]`` is the frame the row was
    admitted at, so a new row never evicts ring entries it did not write.
    """

    def __init__(self, B: int, n_vq: int, vocab: int, window: int, device):
        if window <= 0:
            raise ValueError("repetition window must be > 0")
        self.window = window
        self.counts = torch.zeros(B, n_vq, vocab, dtype=torch.int32, device=device)
        self.ring = torch.zeros(B, n_vq, window, dtype=torch.long, device=device)
        self.start = torch.zeros(B, dtype=torch.long, device=device)
        self._ones = torch.ones(B, 1, dtype=torch.int32, device=device)

    def reset_row(self, b: int, frame: torch.Tensor) -> None:
        self.counts[b] = 0
        self.ring[b] = 0
        self.start[b] = frame

    def penalise(self, logits: torch.Tensor, ch: int, penalty: torch.Tensor) -> torch.Tensor:
        """``logits`` (B, V); ``penalty`` (B, 1). Seen codes: ``<0 -> *p``, ``>=0 -> /p``."""
        bent = torch.where(logits < 0, logits * penalty, logits / penalty)
        return torch.where(self.counts[:, ch] > 0, bent, logits)

    def add(self, ch: int, code: torch.Tensor, frame: torch.Tensor) -> None:
        """Record ``code`` (B,) produced on codebook ``ch`` at global ``frame``."""
        counts = self.counts[:, ch]
        slot = torch.remainder(frame, self.window).view(1)
        old = self.ring[:, ch].index_select(1, slot)                                  # (B, 1)
        evict = ((frame - self.start) >= self.window).to(torch.int32).view(-1, 1)
        counts.scatter_add_(1, old, -(self._ones * evict))
        self.ring[:, ch].index_copy_(1, slot, code.view(-1, 1))
        counts.scatter_add_(1, code.view(-1, 1), self._ones)


def sample_rows(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor,
                top_p: torch.Tensor) -> torch.Tensor:
    """Top-k + top-p sampling with per-row settings, no Python branches (design §3.3).

    ``logits`` (B, V) float; ``temperature``/``top_p`` (B, 1) float; ``top_k`` (B, 1)
    long, ``<= 0`` disables it; ``temperature <= 0`` is greedy. Returns codes (B,).
    """
    V = logits.shape[-1]
    # Empty slots step on stale state; keep multinomial well-defined whatever they make.
    logits = torch.nan_to_num(logits)
    greedy = logits.argmax(dim=-1)
    scaled = logits / temperature.clamp(min=1e-6)
    s_logits, s_idx = torch.sort(scaled, descending=True, dim=-1)
    kth = s_logits.gather(1, (top_k - 1).clamp(0, V - 1))
    kth = torch.where(top_k > 0, kth, torch.full_like(kth, float("-inf")))
    drop = F.softmax(s_logits, dim=-1).cumsum(dim=-1) > top_p
    drop[..., 1:] = drop[..., :-1].clone()     # keep the token that crosses top_p
    drop[..., 0] = False
    drop |= s_logits < kth
    kept = torch.full_like(scaled, float("-inf")).scatter_(-1, s_idx, s_logits.masked_fill(drop, float("-inf")))
    sampled = torch.multinomial(F.softmax(kept, dim=-1), num_samples=1).squeeze(-1)
    return torch.where(temperature.view(-1) <= 0, greedy, sampled)


class StaticBackbone:
    """Qwen3 decode step over a preallocated ring KV cache ``[L, B, n_kv, max_len, hd]``.

    Every row writes its new token at the shared index ``cur % max_len``; rows
    differ only by their attention bias and rotary position. The HF modules do
    the arithmetic; this class owns the cache, bias and positions.
    """

    def __init__(self, model, B: int, max_len: int):
        bb = model.semantic_backbone
        cfg = bb.config
        p = next(bb.parameters())
        self.layers, self.norm, self.rotary = bb.layers, bb.norm, bb.rotary_emb
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.B, self.max_len = B, max_len
        self.k = torch.zeros(len(self.layers), B, self.n_kv, max_len, self.hd, device=p.device, dtype=p.dtype)
        self.v = torch.zeros_like(self.k)
        # Float 0/-inf bias, not a bool mask: keeps SDPA on the efficient kernel.
        self.bias = torch.full((B, 1, 1, max_len), float("-inf"), device=p.device, dtype=p.dtype)
        self.cur = torch.zeros((), device=p.device, dtype=torch.long)
        self.pos = torch.zeros(B, 1, device=p.device, dtype=torch.long)

    def reset(self) -> None:
        self.cur.zero_()
        self.pos.zero_()
        self.bias.fill_(float("-inf"))

    @torch.no_grad()
    def load_row(self, b: int, keys: List[torch.Tensor], values: List[torch.Tensor]) -> None:
        """Write a prefilled prompt (per layer ``(n_kv, T, hd)``) into slot ``b``,
        right-aligned to the write index: positions ``[cur-T, cur)`` mod ``max_len``."""
        T = keys[0].shape[1]
        idx = torch.remainder(self.cur - T + torch.arange(T, device=self.cur.device), self.max_len)
        for i in range(len(self.layers)):
            self.k[i, b].index_copy_(1, idx, keys[i].to(self.k.dtype))
            self.v[i, b].index_copy_(1, idx, values[i].to(self.v.dtype))
        self.bias[b].fill_(float("-inf"))
        self.bias[b, 0, 0].index_fill_(0, idx, 0.0)
        self.pos[b] = T

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B, 1, H) -> last hidden (B, H); advances ``cur`` and ``pos``."""
        B, g = self.B, self.n_heads // self.n_kv
        idx = torch.remainder(self.cur, self.max_len).view(1)
        self.bias.view(B, self.max_len).index_fill_(1, idx, 0.0)      # attend to the new token
        cos, sin = self.rotary(x, self.pos)
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(B, 1, self.n_heads, self.hd)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(B, 1, self.n_kv, self.hd)).transpose(1, 2)
            v = a.v_proj(h).view(B, 1, self.n_kv, self.hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.k[i][:, :, idx] = k
            self.v[i][:, :, idx] = v
            # GQA: fold the g query heads sharing a KV head into the q_len axis
            # (exact; ``enable_gqa`` falls back to the slow math path).
            attn = F.scaled_dot_product_attention(
                q.view(B, self.n_kv, g, self.hd), self.k[i], self.v[i],
                attn_mask=self.bias, scale=a.scaling,
            )
            x = x + a.o_proj(attn.reshape(B, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        self.cur += 1
        self.pos += 1
        return self.norm(x)[:, 0]


class StreamFrame:
    """The captured frame graph over ``B`` slots.

    Device state read by the graph:
      ``h`` (B, H)            backbone hidden feeding the acoustic decoder
      ``codes`` (R, B, n_vq)  ring of the last ``R`` frames' codes, at ``frame % R``
      ``finished`` (B,)       slot empty or row done (EOS or cap)
      ``length`` (B,)         global frame count at which the row finished
      ``caps`` (B,)           global frame count at which the row is cut
      ``frame`` ()            global frame counter (== the scheduler's tick)
      sampling (B, 1)         temperature / top_k / top_p / penalty per row
      ``spk`` (B, D)          speaker x-vector re-added at every step (None if unused)
    """

    def __init__(self, model, B: int, max_len: int, *, rep_window: int = 64,
                 codes_ring: int = 64, warmup: int = 3):
        cfg = model.config
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
        self.model, self.cfg = model, cfg
        self.B, self.max_len, self.ring = B, max_len, codes_ring
        self.n_vq = cfg.n_vq
        self.bb = StaticBackbone(model, B, max_len)
        self.rep = RepHistory(B, cfg.n_vq, cfg.audio_vocab_size, rep_window, dev)
        self.h = torch.zeros(B, cfg.hidden_size, device=dev, dtype=dt)
        self.codes = torch.zeros(codes_ring, B, cfg.n_vq, device=dev, dtype=torch.long)
        self.finished = torch.ones(B, device=dev, dtype=torch.bool)
        self.length = torch.zeros(B, device=dev, dtype=torch.long)
        self.caps = torch.zeros(B, device=dev, dtype=torch.long)
        self.frame = torch.zeros((), device=dev, dtype=torch.long)
        self.temperature = torch.ones(B, 1, device=dev)
        self.top_k = torch.zeros(B, 1, device=dev, dtype=torch.long)
        self.top_p = torch.ones(B, 1, device=dev)
        self.penalty = torch.ones(B, 1, device=dev)
        spk_dim = getattr(cfg, "speaker_embedding_dim", None)
        self.spk = (torch.zeros(B, spk_dim, device=dev, dtype=torch.float32)
                    if model.xvec_proj is not None and spk_dim else None)
        # Constant inputs of the acoustic decoder, allocated once outside the graph.
        self._sgs_ids = torch.full((B,), cfg.speech_generation_start_token_id, device=dev, dtype=torch.long)
        self._ac_pos = [torch.arange(2, device=dev)] + [
            torch.tensor([ch + 1], device=dev) for ch in range(1, cfg.n_vq)]
        self._next_row = torch.full((B, 1, cfg.n_vq + 1), cfg.audio_pad_token_id, device=dev, dtype=torch.long)
        self._next_row[:, :, 0] = cfg.speech_generation_start_token_id

        self.graph = self._capture(warmup)
        self.reset()

    def _capture(self, warmup: int) -> torch.cuda.CUDAGraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            self._body()
        return graph

    @torch.no_grad()
    def reset(self) -> None:
        """Empty every slot (after capture, whose warm-up dirtied the state)."""
        self.h.zero_()
        self.codes.zero_()
        self.finished.fill_(True)
        self.length.zero_()
        self.caps.zero_()
        self.frame.zero_()
        self.rep.counts.zero_()
        self.rep.ring.zero_()
        self.rep.start.zero_()
        self.bb.reset()

    # ── graph body ──────────────────────────────────────────────────────────

    def _sample(self, ch: int, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.model.audio_lm_heads[ch](hidden).float()
        logits = self.rep.penalise(logits, ch, self.penalty)
        code = sample_rows(logits, self.temperature, self.top_k, self.top_p)
        self.rep.add(ch, code, self.frame)
        return code

    def _acoustic(self):
        """One acoustic frame for all slots -> ``(codes (B, n_vq), is_eos (B,))``."""
        m, cfg = self.model, self.cfg
        dec = m.acoustic_decoder
        L = len(dec.layers)
        B, H = self.B, cfg.hidden_size
        dt = next(dec.parameters()).dtype
        tok = torch.stack([self.h.to(dt), m.text_embeddings(self._sgs_ids).to(dt)], dim=1)   # (B, 2, H)
        hidden, pk, pv = dec.cached_step(tok, self._ac_pos[0], [None] * L, [None] * L)
        is_eos = m.text_lm_head(hidden[:, 0]).float().argmax(-1) == cfg.speech_generation_end_token_id
        codes = [self._sample(0, hidden[:, 1])]
        for ch in range(1, self.n_vq):
            emb = m.audio_embeddings[ch - 1](codes[-1]).to(dt).view(B, 1, H)
            hidden, pk, pv = dec.cached_step(emb, self._ac_pos[ch], pk, pv)
            codes.append(self._sample(ch, hidden[:, 0]))
        return torch.stack(codes, dim=1), is_eos

    def _body(self) -> None:
        codes, is_eos = self._acoustic()
        self.codes.index_copy_(0, torch.remainder(self.frame, self.ring).view(1), codes.unsqueeze(0))
        n = self.frame + 1
        done = ~self.finished & (is_eos | (n >= self.caps))      # EOS frame is kept
        self.length.copy_(torch.where(done, n, self.length))
        self.finished.logical_or_(done)
        self.frame += 1
        # Next backbone input: [<sgs>, code_0..code_15] + speaker anchor.
        row = self._next_row.clone()
        row[:, 0, 1:] = codes
        x = self.model._build_inputs_embeds(row, speaker_emb=self.spk)
        self.h.copy_(self.bb.step(x))

    # ── host side ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def admit(self, b: int, f0: int, keys: List[torch.Tensor], values: List[torch.Tensor],
              h_last: torch.Tensor, spk: Optional[torch.Tensor], cap: int, *,
              temperature: float, top_k: int, top_p: float, repetition_penalty: float) -> None:
        """Fill slot ``b`` with a prefilled prompt; its first frame is global frame ``f0``."""
        T = keys[0].shape[1]
        if T + cap + 2 > self.max_len:     # the ring write index must not re-enter the row
            raise ValueError(f"prompt ({T} tokens) + {cap} frames exceeds the {self.max_len}-token cache")
        self.bb.load_row(b, keys, values)
        self.h[b].copy_(h_last.to(self.h.dtype))
        self.rep.reset_row(b, self.frame)
        self.temperature[b] = float(temperature)
        self.top_k[b] = int(top_k)
        self.top_p[b] = float(top_p)
        self.penalty[b] = float(repetition_penalty)
        if self.spk is not None:
            if spk is None:
                raise ValueError("this model needs a speaker embedding for every row")
            self.spk[b].copy_(spk.view(-1).to(self.spk.dtype))
        self.caps[b] = f0 + cap
        self.length[b] = 0
        self.finished[b] = False

    def replay(self) -> None:
        self.graph.replay()

    def status(self):
        """``(finished, length)`` as host lists, in one device->host read."""
        st = torch.stack([self.finished.long(), self.length]).tolist()
        return st[0], st[1]

    def recent_codes(self, tick: int, k: int) -> torch.Tensor:
        """Codes of global frames ``[tick-k, tick)`` for every slot: ``(k, B, n_vq)`` on the device."""
        if k > self.ring:
            raise ValueError(f"asked for {k} frames, ring holds {self.ring}")
        idx = torch.arange(tick - k, tick, device=self.codes.device) % self.ring
        return self.codes.index_select(0, idx)


def check_sampling(temperature: float, top_k: int, top_p: float, repetition_penalty: float) -> None:
    """Reject settings that would put NaN/inf into ``multinomial`` (kills the CUDA context)."""
    if not (math.isfinite(temperature) and temperature >= 0):
        raise ValueError(f"temperature must be finite and >= 0, got {temperature}")
    if int(top_k) != top_k or top_k < 0:
        raise ValueError(f"top_k must be an integer >= 0, got {top_k}")
    if not (math.isfinite(top_p) and 0 < top_p <= 1):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if not (math.isfinite(repetition_penalty) and repetition_penalty > 0):
        raise ValueError(f"repetition_penalty must be finite and > 0, got {repetition_penalty}")
