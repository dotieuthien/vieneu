"""Render docs/optimize.mp4 (+ .srt, poster): the codec optimisation, ~3 min, straight to the point.

    python docs/video/make_optimize_video.py --voice http://127.0.0.1:PORT   # đọc lời bằng server lite (bản tối ưu)
    python docs/video/make_optimize_video.py                                 # dùng giọng đã cache

Same style and voice cache as make_video.py (docs/video/.voice/). Every number on screen comes
from README "Tối ưu codec (14/10/2026)" and the result files it links.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

import make_video as mv
from make_video import (BG, BLUE, F_B, F_H, F_S, F_SUB, F_TITLE, F_XS, GREEN, INK, LINE, MUTED, ORANGE,
                        PURPLE, RED, SR, TEAL, W, H, FPS, arrow, chip, ctext, ease, font, mix, play_icon, rbox, seg,
                        text, tint, wave, wrap)

OUT = Path(__file__).resolve().parents[1]
NAME = "optimize"
# Cách viết để TTS đọc đúng; phụ đề giữ chữ gốc.
mv.SPOKEN.update({"key value": "ki va liu", "megabyte": "mê ga bai", "Profiler": "Bộ đo hiệu năng",
                  "codec": "cô đéc", "Codec": "Cô đéc", "model": "mô hình", "Model": "Mô hình", "code cũ": "mã cũ",
                  "0,9999993": "gần bằng một", "KV cache": "bộ nhớ ki va liu", "transformer": "trans pho mơ",
                  "tính attention": "tính độ chú ý"})
F_MONO = font(20, uni=True)
F_BIG = font(56, True)


def appear(p, a, b=None):
    return ease(seg(p, a, a + 0.12 if b is None else b))


def bullet_lines(d, x, y, lines, alphas, f=F_B, color=INK, gap=44, dot=BLUE):
    for i, (ln, a) in enumerate(zip(lines, alphas)):
        if a <= 0:
            continue
        cy = y + i * gap
        d.ellipse([x, cy + 9, x + 12, cy + 21], fill=mix(dot, a))
        text(d, x + 26, cy, ln, f, mix(color, a))


def hbar(d, x, y, length, h, color, a, label=None, value=None, f=F_B):
    if a <= 0:
        return
    rbox(d, x, y, max(6, length * ease(a)), h, mix(color, a), r=8)
    if value:
        text(d, x + length * ease(a) + 12, y + h / 2, value, f, mix(color, a), anchor="lm")


# ── scenes: fn(d, p, t, ci) — p = 0..1 through the scene, ci = index of the current cue ──────────

CHECK, CROSS = "✓", "✗"
F_SYM = font(30, uni=True)
LANE_BG = (232, 234, 238)
Q = 0.0          # progress 0..1 through the current cue (render_frame sets it)
STAGES = [(4, 4, "10 s", 500), (8, 2, "8 s", 800), (16, 2, "6 s", 1200), (32, 4, "4 s", 1600)]
#          tokens/frame, layers, look-back window, window in tokens   (MOSS-Audio-Tokenizer-Nano decoder)


def source(d, s, a=1.0):
    """Where the number on screen comes from (small grey line under the title)."""
    text(d, 62, 104, "Số liệu: " + s, F_XS, mix(MUTED, a))


def lanes(d, x, y, w, cpu, gpu, title=None):
    """Two time lanes. ``cpu``/``gpu``: [(start, end, colour, label)] in fractions of ``w``."""
    if title:
        text(d, x - 90, y - 34, title, F_B)
    for k, (name, segs, col) in enumerate((("CPU", cpu, INK), ("GPU", gpu, BLUE))):
        yy = y + k * 52
        text(d, x - 90, yy + 8, name, F_B, col)
        rbox(d, x, yy, w, 42, LANE_BG, r=8)
        for s0, s1, c, lab in segs:
            sw = max(4, (s1 - s0) * w - 2)
            rbox(d, x + s0 * w, yy + 3, sw, 36, c, r=6)
            if lab and sw > d.textlength(lab, font=F_XS) + 8:
                ctext(d, x + s0 * w + sw / 2, yy + 21, lab, F_XS, (255, 255, 255))


def old_codec_lanes():
    """One codec call before the fix: per frame the CPU issues ~900 ops (the GPU picks them up
    in slivers), then waits for the GPU to finish so it can read the audio lengths back."""
    cpu, gpu = [], []
    for i in range(4):
        s = i / 4
        cpu.append((s, s + 0.19, ORANGE, f"ra lệnh mẩu {i + 1}"))
        cpu.append((s + 0.19, s + 0.25, RED, "chờ"))
        for j in range(6):
            g = s + 0.015 + j * 0.03
            gpu.append((g, g + 0.008, BLUE, None))
        gpu.append((s + 0.19, s + 0.24, BLUE, None))
    return cpu, gpu


def s_title(d, p, t, ci):
    ctext(d, W / 2, 230, "Tối ưu codec của VieNeu lite", font(52, True))
    ctext(d, W / 2, 305, "đo → tìm chỗ chậm → sửa → kiểm tra", F_H, MUTED)
    for i, c in enumerate([BLUE, ORANGE, GREEN]):
        wave(d, 290, 440, 700, 60, mix(c, 0.85), phase=t * 2 + i * 2.1, n=80)


def s_measure(d, p, t, ci):
    source(d, "profiler, 1 người dùng, 16 ghế, bản cũ · 54 ms = cộng thời gian mọi kernel GPU của codec trong trace")
    x0, scale = 420, 740 / 320            # px per ms; 320 ms of audio = the real-time budget
    text(d, 100, 145, "Mỗi lần gọi codec = 4 mẩu = 320 ms âm thanh", F_B, MUTED)
    d.line([x0, 185, x0, 470], fill=LINE, width=2)
    bx = x0 + 320 * scale
    d.line([bx, 185, bx, 470], fill=RED, width=3)
    text(d, bx - 6, 474, "320 ms", F_XS, RED, anchor="ra")
    rows = [("Model tạo 4 mẩu", 40, BLUE, "≈ 40 ms", 0), ("Codec: CPU bận", 182, ORANGE, "182 ms", 1),
            ("Codec: GPU thật sự tính", 54, GREEN, "≈ 54 ms", 1)]
    for i, (lab, ms, col, val, cue) in enumerate(rows):
        a = 1.0 if ci > cue else (appear(Q, 0.1 + 0.3 * max(0, i - 1)) if ci == cue else 0)
        y = 205 + i * 85
        text(d, 100, y + 8, lab, F_B, mix(INK, a))
        hbar(d, x0, y, ms * scale, 46, col, a, value=val)
    if ci >= 1:
        ctext(d, W / 2, 530, "→ codec bị CPU giữ chân: GPU ngồi chờ lệnh", F_H, mix(RED, appear(Q, 0.6)))


def s_flow(d, p, t, ci):
    source(d, "cấu hình decoder của MOSS-Audio-Tokenizer-Nano (codec của VieNeu v3 Turbo)")
    # input: 4 frames x 16 codes for one user
    text(d, 70, 140, "Đầu vào (1 người, 1 lần gọi)", F_S, MUTED)
    for f in range(4):
        text(d, 70, 178 + f * 34, f"mẩu {f + 1}", F_XS, INK)
        for k in range(16):
            rbox(d, 140 + k * 13, 176 + f * 34, 11, 24, tint(BLUE, 0.35), outline=BLUE, r=2)
    text(d, 70, 318, "16 số / mẩu, mỗi số 0–1023", F_XS, BLUE)
    if ci >= 1:
        arrow(d, 360, 230, 410, 230, MUTED, w=3, head=10)
        rbox(d, 415, 190, 120, 80, tint(PURPLE, 0.14), outline=PURPLE, r=10)
        ctext(d, 475, 218, "1 vector", F_S, PURPLE)
        ctext(d, 475, 245, "768 số / mẩu", F_XS, MUTED)
        arrow(d, 540, 230, 580, 230, MUTED, w=3, head=10)
        for i, (n, layers, win, _) in enumerate(STAGES):
            a = appear(Q, 0.15 + 0.12 * i)
            y = 150 + i * 62
            text(d, 590, y + 4, f"tầng {i + 1}", F_S, mix(INK, a))
            cw = 320 / n
            for k in range(n):
                rbox(d, 680 + k * cw, y, max(3, cw - 2), 30, mix(tint(ORANGE, 0.3), a), outline=mix(ORANGE, a), r=2)
            text(d, 1010, y + 4, f"{n} bước · {layers} lớp", F_XS, mix(INK, a))
            text(d, 1010, y + 26, f"nhìn lại {win}", F_XS, mix(MUTED, a))
        text(d, 680, 400, "1 mẩu được chia dần thành 4 → 8 → 16 → 32 bước thời gian", F_XS, ORANGE)
    if ci >= 2:
        arrow(d, 840, 430, 840, 470, MUTED, w=3, head=10)
        wave(d, 690, 510, 300, 50, GREEN, phase=t * 3)
        text(d, 1000, 490, "32 × 240 = 7680 mẫu", F_S, GREEN)
        text(d, 1000, 518, "= 2 kênh × 3840 = 80 ms", F_S, GREEN)


def s_seq(d, p, t, ci):
    source(d, "trace một lần gọi codec: ~3500 phép tính nhỏ cho 4 mẩu · GPU có việc ≈ 54 / 182 ≈ 30% thời gian")
    cpu, gpu = old_codec_lanes()
    lanes(d, 200, 190, 960, cpu, gpu, title="Code cũ, một lần gọi codec:")
    text(d, 200, 320, "cam = chạy cả 12 lớp cho 1 mẩu (~900 phép, CPU ra lệnh từng phép)", F_S, ORANGE)
    text(d, 200, 355, "đỏ = đọc độ dài âm thanh về CPU → phải đợi GPU làm xong", F_S, RED)
    if ci >= 1:
        ctext(d, W / 2, 450, "4 lượt nối tiếp · GPU chỉ có việc ~30% thời gian", F_H, mix(RED, appear(Q, 0.4)))


def s_kv(d, p, t, ci):
    if ci == 0:
        source(d, "mỗi lớp nhìn lại 500 / 800 / 1200 / 1600 bước tùy tầng (10 / 8 / 6 / 4 giây)")
        text(d, 100, 145, "Một lớp ở tầng 4, một người dùng:", F_B)
        n = 40
        for k in range(n):
            col = tint(PURPLE, 0.35) if k < n - 4 else ORANGE
            rbox(d, 100 + k * 26, 200, 22, 60, col, r=4)
        text(d, 100, 270, "← 1600 bước gần nhất (4 giây): key + value lưu sẵn = KV cache", F_S, PURPLE)
        text(d, 100 + (n - 4) * 26, 270, "bước mới", F_S, ORANGE)
        a = appear(Q, 0.4)
        arrow(d, 100 + (n - 4) * 26, 320, 110, 320, mix(ORANGE, a), w=4, head=14)
        ctext(d, 100 + (n - 4) * 13, 345, "bước mới nhìn lại tối đa 1600 bước (attention)", F_S, mix(ORANGE, a))
        text(d, 100, 400, "Key/value của bước cũ đã lưu sẵn → không phải tính lại quá khứ.", F_B, mix(INK, a))
        return
    source(d, "ước tính: 16 ghế × 12 lớp × ~1000 bước × 512 số (256 key + 256 value) × 4 byte · ghi lại: đọc mã nguồn")
    terms = [("16", "ghế"), ("12", "lớp"), ("~1000", "bước"), ("512", "số"), ("4", "byte")]
    for i, (v, lab) in enumerate(terms):
        x = 100 + i * 150
        rbox(d, x, 150, 120, 90, (255, 255, 255), outline=PURPLE, r=12)
        ctext(d, x + 60, 180, v, F_H, PURPLE)
        ctext(d, x + 60, 218, lab, F_S, MUTED)
        if i < 4:
            ctext(d, x + 135, 195, "×", F_H, MUTED)
    text(d, 860, 172, "≈ 400 MB", F_BIG, PURPLE)
    text(d, 100, 285, "Code cũ, mỗi lớp, mỗi lượt:", F_B)
    for i, s in enumerate(["1. nối bước mới vào", "2. cắt bỏ bước quá cũ", "3. chọn ghế được tính", "4. chép ngược về"]):
        a = appear(Q, 0.15 + 0.12 * i)
        x = 100 + i * 270
        rbox(d, x, 330, 250, 60, mix(tint(RED, 0.10), a), outline=mix(RED, a), r=10)
        ctext(d, x + 125, 360, s, F_S, mix(RED, a))
    text(d, 100, 420, "= ghi lại toàn bộ cửa sổ 4 lần, dù chỉ thêm vài bước", F_H, mix(RED, appear(Q, 0.65)))


def s_group(d, p, t, ci):
    x0, y0, cs = 120, 175, 62
    text(d, x0, 140, "bước sau chỉ nhìn bước trước", F_S, MUTED)
    for r in range(4):
        text(d, x0 - 100, y0 + r * cs + 20, f"mẩu {r + 1}", F_XS, INK)
        for c in range(4):
            x, y = x0 + c * cs, y0 + r * cs
            if c <= r:
                rbox(d, x + 3, y + 3, cs - 6, cs - 6, tint(GREEN, 0.3), outline=GREEN, r=8)
                ctext(d, x + cs / 2, y + cs / 2, CHECK, F_SYM, GREEN)
            else:
                rbox(d, x + 3, y + 3, cs - 6, cs - 6, tint(RED, 0.12), outline=RED, r=8)
                ctext(d, x + cs / 2, y + cs / 2, CROSS, F_SYM, RED)
    text(d, 470, 180, "Đưa cả 4 mẩu vào một lượt:", F_H)
    text(d, 470, 225, "ô đỏ chặn mẩu trước nhìn mẩu sau,", F_B)
    text(d, 470, 260, "nên ra đúng như chạy lần lượt từng mẩu.", F_B)
    if ci >= 1:
        rows = [("lượt chạy 12 lớp", "4", "1"), ("ghi lại cả cửa sổ KV", "16 lần", "4 lần"),
                ("đọc độ dài về CPU", "mỗi mẩu", "bỏ (luôn 320 ms)")]
        text(d, 470, 330, "", F_S)
        text(d, 780, 330, "cũ", F_S, MUTED, anchor="ma")
        text(d, 990, 330, "mới", F_S, GREEN, anchor="ma")
        for i, (lab, old, new) in enumerate(rows):
            a = appear(Q, 0.1 + 0.15 * i)
            y = 365 + i * 50
            text(d, 470, y, lab, F_B, mix(INK, a))
            text(d, 780, y, old, F_B, mix(RED, a), anchor="ma")
            text(d, 990, y, new, F_B, mix(GREEN, a), anchor="ma")


def s_graph(d, p, t, ci):
    source(d, "182 / 54 ms: bản cũ (trace) · 1,3 / 32 ms: gom + CUDA graph, đo trong ablation, cùng 16 ghế")
    cpu, gpu = old_codec_lanes()
    lanes(d, 200, 170, 960, cpu, gpu, title="Trước: CPU bận 182 ms")
    lanes(d, 200, 360, 960, [(0, 0.008, GREEN, None)], [(0.008, 0.008 + 32 / 182, BLUE, "4 mẩu")],
          title="Sau (gom + CUDA graph): CPU ~1,3 ms, GPU ~32 ms")
    text(d, 200, 475, "CUDA graph: ~900 phép của một lượt → 1 lệnh phát lại; GPU vẫn làm đủ việc", F_B, BLUE)


def ring_cells(d, x, y, labels, written, cw=78, color=PURPLE, dim=()):
    for k, lab in enumerate(labels):
        w_on = k in written
        fill = ORANGE if w_on else ((235, 236, 240) if k in dim else tint(color, 0.25))
        rbox(d, x + k * (cw + 6), y, cw, 54, fill, outline=ORANGE if w_on else color, r=8)
        ctext(d, x + k * (cw + 6) + cw / 2, y + 27, lab, F_S, (255, 255, 255) if w_on else INK)


def s_ring(d, p, t, ci):
    if ci == 0:
        text(d, 100, 140, "Cách cũ: cửa sổ giữ đúng thứ tự thời gian (ví dụ cửa sổ 8 bước)", F_B)
        ring_cells(d, 100, 190, [f"b{k}" for k in range(1, 9)], set())
        text(d, 100, 255, "trước", F_XS, MUTED)
        a = appear(Q, 0.35)
        if a > 0:
            text(d, 820, 205, "+ b9, b10 mới", F_B, mix(ORANGE, a))
            ring_cells(d, 100, 300, [f"b{k}" for k in range(3, 11)], set(range(8)) if a > 0.5 else set())
            text(d, 100, 365, "sau: dời cả cửa sổ → ghi lại 8/8 ô", F_S, RED)
        text(d, 100, 430, "Tầng 4: ghi lại cả 1600 bước, 4 lần mỗi lớp, mỗi lượt", F_H, mix(RED, appear(Q, 0.7)))
        return
    if ci == 1:
        source(d, "độ dài vòng = cửa sổ + 4 mẩu; tầng 4: 1600 + 128 bước")
        text(d, 100, 140, "Cách mới: mảng cố định 10 ô = cửa sổ 8 + 2 bước mới; ô = số thứ tự bước mod 10", F_B)
        ring_cells(d, 100, 190, ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"], set(), cw=70)
        a = appear(Q, 0.35)
        if a > 0:
            text(d, 100, 260, "+ b11, b12 mới: ghi đè vào ô của b1, b2 (đã ra khỏi cửa sổ)", F_B, mix(ORANGE, a))
            ring_cells(d, 100, 300, ["b11", "b12", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"],
                       {0, 1} if a > 0.5 else set(), cw=70)
            text(d, 100, 365, "chỉ ghi 2/10 ô, các ô khác giữ nguyên", F_S, GREEN)
        return
    source(d, "tầng 4, mỗi lần gọi codec (4 mẩu × 32 bước): cũ ghi lại cửa sổ 1600 bước, mới ghi 128 bước mới")
    text(d, 100, 140, "Khi tính attention, thứ tự được tính lại từ số thứ tự bước:", F_B)
    ring_cells(d, 100, 190, ["b11", "b12", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"], set(), cw=70,
               dim={2, 3} if Q > 0.3 else ())
    text(d, 100, 255, "b3, b4: quá 8 bước so với b12 → che đi (xám); kết quả y hệt cách cũ", F_S, MUTED)
    rows = [("Ghi mỗi lần gọi (tầng 4)", "1600 bước × 4 lần", "128 bước mới")]
    for i, (lab, old, new) in enumerate(rows):
        y = 330
        text(d, 100, y, lab, F_B)
        text(d, 560, y, old, F_B, RED)
        text(d, 900, y, new, F_B, GREEN)
    text(d, 560, 300, "chưa có vòng (đã gom)", F_S, MUTED)
    text(d, 900, 300, "mới", F_S, GREEN)


def s_verify(d, p, t, ci):
    source(d, "CODEC_VERIFY=1 (chạy mỗi lần khởi động) · mẫu nghe so sánh: docs/samples/")
    text(d, 100, 145, "Cùng một chuỗi mã → codec cũ và codec mới", F_H)
    for k, col in enumerate([BLUE, ORANGE]):
        wave(d, 120, 270, 1040, 120, mix(col, 0.9 if k == 0 else 0.7), phase=1.3, n=160, width=4 - k * 2)
    text(d, 120, 340, "xanh = cũ · cam = mới (trùng lên nhau)", F_S, MUTED)
    rbox(d, 380, 390, 520, 120, tint(GREEN, 0.12), outline=GREEN, r=16)
    ctext(d, 640, 420, "tương quan (1 = trùng khít)", F_S, MUTED)
    ctext(d, 640, 470, "0,9999993", F_BIG, GREEN)


PARAMS = [
    ("4 mẩu mỗi lần gọi codec", "mỗi lượt đọc lại KV cache của mọi ghế → gom lại cho rẻ; mỗi đoạn vẫn chỉ 320 ms"),
    ("người mới: sau 2 mẩu", "giải mã sớm để tiếng đầu ra nhanh; lần sớm này chỉ tính cho người mới"),
    ("số ghế = số người dự kiến", "mỗi lượt luôn tính đủ mọi ghế, kể cả ghế trống"),
]


def s_params(d, p, t, ci):
    for i, (name, why) in enumerate(PARAMS):
        a = appear(Q, 0.05 + 0.3 * i)
        y = 160 + i * 110
        rbox(d, 90, y, 1100, 90, mix((255, 255, 255), a), outline=mix(BLUE, a), r=14)
        text(d, 115, y + 30, name, F_B, mix(BLUE, a))
        lines = wrap(why, F_S, 620, d)[:2]
        for k, ln in enumerate(lines):
            text(d, 540, y + (31 if len(lines) == 1 else 18 + k * 28), ln, F_S, mix(INK, a))


ABLATION = [   # lite/results/ablation-20261015/summary.json: (name, RTF 1, RTF 16, TTFA 1, TTFA 16, codec CPU / GPU)
    ("Bản cũ", "0,40", "0,54", "124", "294", "—"),
    ("+ gom mẩu", "0,21", "0,32", "111", "325", "40 ms / —"),
    ("+ gom + graph", "0,21", "0,31", "88", "338", "1,3 / 32 ms"),
    ("+ gom + vòng", "0,18", "0,29", "104", "340", "34 ms / —"),
    ("+ cả ba", "0,175", "0,25", "76", "300", "1,2 / 22 ms"),
]
ABL_COLS = [("RTF 1 người", 470), ("RTF 16 người", 610), ("tiếng đầu, 1 người", 765), ("tiếng đầu, 16 người", 925),
            ("codec CPU / GPU", 1090)]
ABL_FOCUS = [({0, 1, 2}, {0, 2, 4}), ({2, 4}, {0, 1, 4}), ({0, 4}, {3})]          # per cue: (rows, cols)
ABL_NOTE = ["gom: RTF 0,40 → 0,21 · graph: tiếng đầu 111 → 88 ms", "vòng: GPU 32 → 22 ms · cả ba: 0,175 / 0,25",
            "đông người: tiếng đầu ≈ 300 ms, như bản cũ"]


def s_ablation(d, p, t, ci):
    source(d, "bật/tắt từng cách sửa trên cùng một máy RTX 3060, 16 ghế, trung vị 8 lượt mỗi mức; tiếng đầu tính bằng ms")
    rows, cols = ABL_FOCUS[ci]
    for name, x in ABL_COLS:
        text(d, x, 140, name, F_XS, MUTED, anchor="ma")
    for i, row in enumerate(ABLATION):
        y = 175 + i * 64
        rbox(d, 80, y, 1120, 54, tint(BLUE, 0.08) if i in rows else (255, 255, 255), outline=LINE, r=12)
        text(d, 100, y + 15, row[0], F_B, INK if i in rows else MUTED)
        for j, ((name, x), v) in enumerate(zip(ABL_COLS, row[1:])):
            hot = i in rows and j in cols
            text(d, x, y + 13, v, F_B if hot else F_S, BLUE if hot else (INK if i in rows else MUTED), anchor="ma")
    text(d, 100, 505, ABL_NOTE[ci], F_H, RED if ci == 2 else GREEN)


SCENES = [
    ("", "", s_title, [
        "Video này đi thẳng vào phần tối ưu codec của bản lite: đo, tìm chỗ chậm, sửa, rồi kiểm tra.",
    ]),
    ("Bước 1", "Đo trước khi sửa", s_measure, [
        "Profiler đo một lần gọi codec, tức 4 mẩu bằng 320 mili giây âm thanh. Model tạo 4 mẩu đó chỉ mất khoảng 40 mili giây.",
        "Còn codec thì CPU bận 182 mili giây, trong khi GPU chỉ thật sự tính khoảng 54 mili giây. Nghĩa là codec bị CPU giữ chân.",
    ]),
    ("Codec", "Codec nhận gì và làm gì", s_flow, [
        "Đầu vào của codec: mỗi mẩu 80 mili giây, model sinh ra 16 con số cho mỗi người dùng. Mỗi lần gọi, codec nhận 4 mẩu như vậy.",
        "Codec gộp 16 số của một mẩu thành một vector 768 số, rồi qua 4 tầng transformer, tổng cộng 12 lớp. Qua mỗi tầng, mẩu được chia nhỏ dần thành 4, 8, 16, rồi 32 bước thời gian.",
        "Cuối cùng, mỗi bước cho ra 240 mẫu âm thanh. 32 bước là 7680 mẫu, chia cho 2 kênh, đúng 80 mili giây.",
    ]),
    ("Chỗ chậm 1", "Giải mã từng mẩu một", s_seq, [
        "Code cũ chạy cả 12 lớp cho mẩu 1, đọc độ dài âm thanh về CPU, rồi mới tới mẩu 2. Lệnh đọc này bắt CPU đứng đợi GPU làm xong.",
        "Mỗi lượt khoảng 900 phép tính nhỏ, CPU ra lệnh từng phép. Bốn lượt nối tiếp, nên GPU chỉ có việc khoảng 30 phần trăm thời gian.",
    ]),
    ("Chỗ chậm 2", "KV cache của codec", s_kv, [
        "Mỗi lớp transformer chỉ nhìn lại một cửa sổ quá khứ, từ 4 tới 10 giây tùy tầng. Key và value của các bước trong cửa sổ được lưu sẵn, gọi là KV cache, để bước mới khỏi tính lại quá khứ.",
        "Với 16 ghế, KV cache của codec khoảng 400 megabyte. Code cũ, mỗi lớp và mỗi lượt, ghi lại toàn bộ cửa sổ 4 lần: nối bước mới, cắt bước cũ, chọn ghế, rồi chép ngược về.",
    ]),
    ("Sửa 1", "Gom 4 mẩu vào một lượt", s_group, [
        "Vì mỗi lớp chỉ nhìn về quá khứ, đưa cả 4 mẩu vào một lượt vẫn ra đúng như chạy lần lượt: mẩu trước không bao giờ thấy mẩu sau.",
        "Một lượt thay cho bốn, nên cửa sổ KV chỉ bị ghi lại 4 lần thay vì 16. Và không cần đọc độ dài về nữa, vì 4 mẩu luôn là 320 mili giây.",
    ]),
    ("Sửa 2", "CUDA graph", s_graph, [
        "CUDA graph ghi lại khoảng 900 phép tính của một lượt thành một lệnh duy nhất. CPU của codec giảm xuống khoảng 1,3 mili giây mỗi lần gọi, GPU làm một mạch khoảng 32 mili giây.",
    ]),
    ("Sửa 3", "KV cache dạng vòng", s_ring, [
        "Cách cũ giữ cửa sổ theo đúng thứ tự thời gian. Thêm bước mới thì phải dời cả cửa sổ, tức ghi lại mọi ô.",
        "Cách mới dùng một mảng cố định, dài bằng cửa sổ cộng thêm 4 mẩu. Bước mới ghi đè vào ô của bước cũ nhất đã ra khỏi cửa sổ, các ô khác giữ nguyên.",
        "Khi tính attention, thứ tự được tính lại từ số thứ tự bước, nên kết quả y hệt. Ở tầng 4, mỗi lần gọi chỉ ghi 128 bước mới, thay vì ghi lại cả cửa sổ 1600 bước.",
    ]),
    ("Kiểm tra", "Âm thanh không đổi", s_verify, [
        "Cùng một chuỗi mã cho qua codec cũ và codec mới: tương quan 0,9999993, nghe không khác.",
    ]),
    ("Tham số", "Chọn tham số", s_params, [
        "Gọi codec mỗi 4 mẩu, vì mỗi lượt phải đọc lại KV cache của mọi ghế. Người mới được giải mã sớm sau 2 mẩu để tiếng đầu ra nhanh. Số ghế nên bằng số người dùng dự kiến, vì mỗi lượt luôn tính đủ mọi ghế.",
    ]),
    ("Kết quả", "Mỗi cách sửa đóng góp bao nhiêu", s_ablation, [
        "Đo từng phần trên cùng một máy: gom mẩu đưa hệ số thời gian thực từ 0,40 xuống 0,21. CUDA graph giữ hệ số gần như cũ, nhưng tiếng đầu nhanh hơn, từ 111 xuống 88 mili giây.",
        "KV cache dạng vòng giảm việc GPU của codec từ 32 xuống 22 mili giây. Cả ba cộng lại: 0,175 với một người, 0,25 với 16 người.",
        "Chưa cải thiện: khi 16 người vào cùng lúc, tiếng đầu vẫn khoảng 300 mili giây, phần lớn là chờ tới lượt và chờ nạp câu.",
    ]),
]


def build_timeline(voice):
    tl, t0 = [], 0.0
    for label, title, fn, cues in SCENES:
        times, t = [], t0
        for c in cues:
            a = mv.load_voice(c, voice)
            dur = len(a) / SR + mv.PAUSE if a is not None else mv.cue_seconds(c)
            times.append((t, t + dur, c))
            t += dur
        t += mv.SCENE_PAUSE
        tl.append((t0, t, label, title, fn, times))
        t0 = t
    return tl, t0


def render_frame(tl, now):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for t0, t1, label, title, fn, cues in tl:
        if t0 <= now < t1 or (now >= t1 and t1 == tl[-1][1]):
            p = min(1.0, max(0.0, (now - t0) / (t1 - t0)))
            ci = next((i for i, (a, b, _) in enumerate(cues) if a <= now < b), len(cues) - 1)
            global Q
            Q = min(1.0, max(0.0, (now - cues[ci][0]) / (cues[ci][1] - cues[ci][0])))
            if title:
                x = 60
                if label:
                    x += chip(d, 60, 44, label, BLUE, F_S) + 16
                text(d, x, 46, title, F_TITLE)
            fn(d, p, now - t0, ci)
            lines = wrap(cues[ci][2], F_SUB, W - 200, d)
            bh = 22 + 38 * len(lines)
            rbox(d, 70, H - bh - 20, W - 140, bh, (24, 27, 31), r=14)
            for i, ln in enumerate(lines):
                d.text((W / 2, H - bh - 20 + 14 + i * 38), ln, font=F_SUB, fill=(255, 255, 255), anchor="ma")
            break
    return img


def voice_track(tl, total, voice):
    import tempfile
    import wave as wavfile
    import numpy as np
    track = np.zeros(int(total * SR) + SR, dtype=np.int16)
    for *_, cues in tl:
        for a, _, c in cues:
            v = mv.load_voice(c, voice)
            if v is None:
                return None
            i = int(a * SR)
            track[i:i + len(v)] = v
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wavfile.open(f.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(track[:int(total * SR)].tobytes())
    return f.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", metavar="URL", help="lite server to read the subtitles aloud")
    ap.add_argument("--speaker", default=None, help="preset voice name (default: server default)")
    ap.add_argument("--frames", default=None, help="only save PNG stills at these comma-separated seconds")
    args = ap.parse_args()
    if args.voice:
        for *_, cues in SCENES:
            for c in cues:
                mv.fetch_voice(args.voice.rstrip("/"), c, args.speaker)

    tl, total = build_timeline(args.speaker)
    if args.frames:
        for s in args.frames.split(","):
            still = mv.VOICE_DIR.parent / ".preview" / f"{NAME}-{float(s):05.1f}.png"
            still.parent.mkdir(exist_ok=True)
            render_frame(tl, float(s)).save(still)
        return
    audio = voice_track(tl, total, args.speaker)
    with open(OUT / f"{NAME}.vi.srt", "w", encoding="utf-8") as f:
        n = 0
        for *_, cues in tl:
            for a, b, c in cues:
                n += 1
                f.write(f"{n}\n{mv.srt_time(a)} --> {mv.srt_time(b)}\n{c}\n\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "slow", "-tune", "animation",
            "-movflags", "+faststart", str(OUT / f"{NAME}.mp4")]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames = int(total * FPS)
    for i in range(frames):
        ff.stdin.write(render_frame(tl, i / FPS).tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    if audio:
        os.unlink(audio)
    t0, t1 = next((a, b) for a, b, lb, *_ in tl if lb == "Bước 1")
    render_frame(tl, t0 + 0.9 * (t1 - t0)).save(OUT / f"{NAME}.png", optimize=True)
    print(f"{total:.0f}s, {frames} frames, voice-over: {'yes' if audio else 'no'} -> {OUT / f'{NAME}.mp4'}")


if __name__ == "__main__":
    main()
