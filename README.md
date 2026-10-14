# VieNeu-TTS inference

[VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) v3 Turbo là model đọc tiếng Việt chạy được thời gian thực:
gửi một đoạn văn, khoảng 130 ms sau đã nghe được tiếng đầu tiên, phần còn lại được stream tiếp trong lúc model
vẫn đang tạo. Tốc độ đó không nằm ở bản thân model (khá nhỏ), mà ở cách server chạy nó: CUDA graph,
nhiều người dùng chung một lần chạy GPU, và codec giải mã từng đoạn ngắn.

Repo này ghi lại quá trình tìm hiểu phần đó:

1. **Đọc hiểu** đường inference của upstream: model sinh âm thanh thế nào, vì sao cần CUDA graph, scheduler chia GPU cho nhiều request ra sao.
2. **Viết lại một bản đơn giản** (lite) chỉ giữ đường streaming trên GPU, đủ nhỏ để đọc hết trong một buổi.
3. **Chạy thật trên GPU thuê qua [VKong](https://vkong.tli-tech.com)**, do AI agent điều khiển: deploy, fix bugs, gọi API, benchmark, tải log và trả máy.
4. **So sánh** bản lite với SDK gốc bằng cùng một client benchmark.
5. **Tối ưu theo profiler** mà không đụng vào model: codec nhanh hơn, RTF giảm khoảng một nửa, audio giữ nguyên ([chi tiết và mẫu nghe](#tối-ưu-codec-14102026)).

## VieNeu hoạt động như thế nào?

[![Video giải thích VieNeu](docs/how-it-works.png)](docs/how-it-works.mp4)

*Giọng đọc trong video được tạo bằng chính bản lite.*

## Hai bản trong repo

- **[baseline](baseline/)**: gọi SDK VieNeu nguyên bản (`Vieneu(...).infer_stream()`), thêm wrapper HTTP có semaphore giới hạn số stream.
- **[lite](lite/)**: viết lại phần streaming trên GPU; vẫn dùng SDK để load model, codec, phonemizer và giọng preset.

Bản lite gói đường serving trong các file sau:

| File | Việc |
|---|---|
| [frame.py](lite/vieneu_lite/frame.py) | Một CUDA graph cho mỗi mẩu 80 ms: chọn 16 mã, chống lặp, sampling theo từng request, KV cache cấp phát sẵn |
| [scheduler.py](lite/vieneu_lite/scheduler.py) | `MAX_STREAMS` slot dùng chung GPU: nhận request, prefill, chạy graph, gom 4 mẩu rồi gọi codec |
| [codec.py](lite/vieneu_lite/codec.py) | Codec MOSS chạy streaming theo slot: một lần forward cho cả job, CUDA graph, KV cache dạng ring (xem [Tối ưu codec](#tối-ưu-codec-14102026)) |
| [tts.py](lite/vieneu_lite/tts.py) | Cắt văn bản thành câu, đổi sang phoneme, nối các câu với khoảng nghỉ đúng |
| [profiler.py](lite/vieneu_lite/profiler.py) | Profiler bật bằng `PROFILE=1`; tắt thì không tốn gì |

So với SDK gốc, lite bỏ các đường không dùng cho GPU streaming (CPU/ONNX, GGUF, static batch, Gradio) và thêm vài chốt an toàn:
tham số sampling sai bị từ chối trước khi vào GPU, lỗi của một request không làm chết server, `/health` trả 503 khi worker chết.

Cấu hình qua biến môi trường của `serve.py`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MAX_STREAMS` | 16 | Số slot (số request tạo audio cùng lúc); request vượt quá phải xếp hàng |
| `DECODE_FRAMES` / `LEADIN_FRAMES` | 4 / 2 | Gọi codec mỗi 4 mẩu; request mới được giải mã sớm sau 2 mẩu |
| `OVERLAP_CODEC` | 1 | Codec chạy trên thread và CUDA stream riêng, song song với việc tạo mẩu; `0` chạy tuần tự |
| `CODEC_GRAPHS` / `CODEC_RING` | 1 / 1 | Codec chạy bằng CUDA graph / KV cache dạng ring; `0` để quay về bản eager / cache của SDK |
| `CODEC_DTYPE` | `fp32` | `bf16` chạy transformer của codec bằng bf16 (chưa kiểm tra chất lượng) |
| `CODEC_VERIFY` | 0 | `1`: lúc khởi động so codec với đường decode gốc của SDK, kết quả ở `/health` và log |
| `PROFILE` / `PROFILE_DIR` | 0 / `results/profile` | Bật profiler; nơi lưu trace |
| `BENCH_MODE` | 0 | Mở `/benchmark?repeats=N` và `/stress` để đo ngay trên máy GPU |
| `WATERMARK` | 1 | Dấu ẩn đánh dấu giọng AI; các phép đo đều tắt |

`lite/vkong.yaml` hiện bật các cờ đo đạc (`PROFILE=1 BENCH_MODE=1 CODEC_VERIFY=1 MAX_STREAMS=32`); bỏ ba cờ đầu khi chạy thật.

### Profiler (bản lite)

Bật bằng `PROFILE=1`. Khi tắt, các hook không làm gì: decorator trả lại hàm gốc, còn `with` dùng một context rỗng.

```bash
curl -X POST "$SERVE_URL/profile/reset"                  # bắt đầu cửa sổ đo
curl -X POST "$SERVE_URL/benchmark?concurrency=16"       # tạo tải
curl -X POST "$SERVE_URL/profile/trace?seconds=2"        # torch.profiler mọi thread, lưu results/profile/*.json.gz
curl "$SERVE_URL/profile"                                # bảng tổng hợp (?format=json cho máy đọc)
curl -O "$SERVE_URL/results/profile/<file>.json.gz"      # tải trace, kéo thả vào ui.perfetto.dev
```

- **Từng vùng** (worker, codec, các thread request): thời gian host (đã trừ vùng con) và thời gian GPU (CUDA event).
  `tick.period` là thời gian một frame cho tất cả slot (≤ 80 ms mới đạt real time); `tick.gap` là lúc AR stream
  không chạy graph; `decode.backpressure` là lúc worker bị codec chặn; `gil.wait` cho biết các thread tranh GIL.
- **Từng request**: TTFA tách thành queue, admit, lead-in, chờ codec, codec, deliver.
- **Frame anatomy** (in ra log lúc khởi động): chạy frame một lần eager và một lần bằng graph, so số kernel, thời gian
  launch và kernel theo từng phần (acoustic, sample, backbone, attention), cùng khoảng hở giữa các node của graph.
- **Trace**: số lần launch, thời gian trong API launch, thời gian sync và thời gian kernel của từng vùng, kèm kết luận
  launch-bound / GPU-bound; mở file trace bằng [Perfetto](https://ui.perfetto.dev).
- **Watchdog**: nếu có request đang chạy mà không frame nào được tạo trong 30 giây, log in stack của mọi thread.
  GPU treo không để lại lỗi Python nào, nên đây là bằng chứng duy nhất.

Hai chỗ dễ sai khi tự dựng profiler kiểu này, đã xử lý trong `profiler.py`: `torch.profiler` mặc định chỉ ghi thread gọi nó
(worker và codec chạy thread riêng nên phải bật `profile_all_threads`), và kernel bên trong CUDA graph không gắn với
CPU op nào (phải ghép theo correlation id của `cudaGraphLaunch`).

## Chạy trên VKong cùng agent

VKong cho thuê GPU theo giờ và điều khiển hoàn toàn bằng CLI, nên agent tự làm được cả vòng:
viết code → `vkong run` → gọi API qua tunnel localhost → chạy benchmark ngay trên máy GPU → tải log về → `vkong app stop`.
Con người chỉ cần đọc kết quả và quyết định bước tiếp theo. Một RTX 3060 có giá khoảng $0.06–0.10/giờ.

`lite/vkong.yaml` ghi `gpu: RTX-3060-12GB`: tên trần `RTX 3060` cho phép thuê cả bản 8 GB hoặc 3060 Ti, kết quả đo sẽ không so được.

**1. Thuê máy và chạy server.** Chạy trong thư mục bản muốn deploy (`lite` hoặc `baseline`):

```bash
cd lite
vkong validate                       # kiểm tra cấu hình vkong.yaml
vkong run                            # thuê RTX 3060, cài thư viện, chạy serve.py
```

Đợi dòng `Workload running · http://127.0.0.1:PORT`; đó là địa chỉ của server trên máy bạn.

**2. Gọi thử API.** Mở terminal khác, thay `PORT` bằng cổng ở bước 1:

```bash
SERVE_URL=http://127.0.0.1:PORT
curl "$SERVE_URL/health"             # server sẵn sàng chưa
curl "$SERVE_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"input":"Xin chào.","response_format":"pcm"}' \
  --output speech.pcm                # audio PCM 16-bit, mono, 48 kHz
```

**3. Cập nhật code** trên máy đang thuê (không cần thuê lại):

```bash
vkong instances                      # xem <instance-id> (dạng vk_...)
vkong attach <instance-id> --update
```

Những điều gặp phải khi làm việc với VKong (vkong v0.10.26, 14/10/2026):

- Tunnel localhost hay đứt (`Connection to the machine was lost`) trong khi máy vẫn `running`. Đừng vội kết luận server treo:
  kiểm tra `vkong instances`, `vkong attach <id>` lại, và xem log có dòng watchdog `no frame for` không.
  Vì vậy benchmark nên chạy ngay trên máy GPU (`/benchmark`, `/stress`), qua tunnel chỉ hỏi trạng thái và tải file.
- `attach --update` bị từ chối khi `vkong.yaml` có `storage:` (`stop the Storage writer before starting another Run`);
  đã gửi feedback (fb-133). Không dùng volume thì cập nhật code bình thường.
- Nếu lần cài đặt đầu tiên của máy chưa xong, `attach --update` báo `.vkong/init_cmd.sh: No such file` vì
  `.vkongignore` loại thư mục `.vkong/`.
- `vkong run` không tương tác (agent) không in giá mà đòi `--accept-price`; hãy chốt mức giá với người dùng trước.

**4. Trả máy** khi dùng xong. `Ctrl+C` chỉ ngắt kết nối, máy vẫn tính tiền cho tới lệnh này:

```bash
vkong app stop vieneu-serving-lite   # bản baseline: vkong app stop vieneu-original-baseline
```

Ảnh chụp từ dashboard VKong, trong lần chạy lite để tạo giọng đọc cho video:

![Tiến trình chạy app vieneu-serving-lite trên VKong](docs/vkong_app.png)

*Tiến trình chạy: thuê máy 29 giây, đồng bộ code 5 giây, cài thư viện 1 phút 20 giây rồi chạy server.*

![Log server khi đọc lời cho video](docs/logs.png)

*Log server: mỗi câu lời thoại là một request, gửi lần lượt từng câu. Audio đầu tiên ra sau 181–205 ms (đo phía server), RTF 0.66–0.71.*

![Biểu đồ GPU, CPU và RAM của máy](docs/metrics.png)

*Biểu đồ GPU, CPU và RAM của máy trong lần chạy đó.*

## Kết quả (RTX 3060, 13/10/2026)

**TTFA**: thời gian từ lúc gửi yêu cầu đến khi nhận audio đầu tiên (ms). **RTF**: tổng thời gian ÷ độ dài audio.
Cả hai càng thấp càng tốt. Số trong bảng là p50 tính trên từng request.

**Lite (trước khi có overlap) so với baseline**: đo theo thứ tự A–B–B–A, 630 request mỗi bản.

| Request đồng thời | TTFA baseline | TTFA lite | RTF baseline | RTF lite |
|---:|---:|---:|---:|---:|
| 1 | 127.0 | 129.1 | 0.529 | 0.548 |
| 4 | 174.4 | 173.6 | 0.568 | 0.568 |
| 8 | 202.6 | 202.5 | 0.591 | 0.590 |
| 16 | 243.6 | 273.8 | 0.674 | 0.659 |
| 32 (32 slot) | 588.4 | 591.5 | 1.040 | 1.039 |

**Overlap codec**: phép thử nhỏ, mỗi cấu hình 6 request ở mức 1 và 32 request ở mức 16.

| Request đồng thời | Cấu hình | TTFA | RTF |
|---:|---|---:|---:|
| 1 | Baseline | 131.1 | 0.535 |
| 1 | Lite, `OVERLAP_CODEC=0` | 131.7 | 0.552 |
| 1 | Lite, `OVERLAP_CODEC=1` | 151.0 | 0.497 |
| 16 | Baseline | 272.9 | 0.730 |
| 16 | Lite, `OVERLAP_CODEC=0` | 233.9 | 0.680 |
| 16 | Lite, `OVERLAP_CODEC=1` | 319.4 | 0.652 |

**Rút ra:**

- Lite chạy xấp xỉ SDK gốc, không nhanh hơn: phần làm nên tốc độ là thiết kế của tác giả VieNeu, lite chỉ viết lại cho dễ đọc.
- Codec song song: cả câu xong sớm hơn chút nhưng tiếng đầu đến chậm hơn; mẫu còn nhỏ.

Điều kiện chung: client chạy ngay trên máy GPU (không tính tunnel); 16 slot, K=4, leadin=2; tắt watermark
(dấu ẩn đánh dấu giọng AI) ở cả hai bản. Mọi request đọc cùng một câu bằng giọng mặc định Hải Đăng:

> Xin chào, đây là giọng đọc VieNeu phát trực tiếp qua API chuẩn OpenAI.

Đây là câu mặc định của client benchmark upstream, dài khoảng 4 giây audio. Kết quả vì vậy chỉ phản ánh
câu ngắn gửi dồn cùng lúc; văn bản dài nhiều câu hoặc request đến rải rác chưa được đo.
## Tối ưu codec (14/10/2026)

[![Video: tối ưu VieNeu lite](docs/optimize.png)](docs/optimize.mp4)

*Video 3 phút 31 giây: vì sao biết phải sửa codec, từng ý tưởng tối ưu, logic chọn tham số và kết quả. Lồng tiếng bằng
chính bản tối ưu; phụ đề: [optimize.vi.srt](docs/optimize.vi.srt); dựng bằng
[make_optimize_video.py](docs/video/make_optimize_video.py). Lời đọc được PhoWhisper nghe lại để sửa những từ phát âm
sai trước khi dựng (ví dụ đọc "codec" thành "cô đéc", "model" thành "mô hình").*

### Quy trình tối ưu (không đụng vào model)

Nguyên tắc: **không đổi trọng số, kiến trúc, dtype hay cách sampling**. Model AR và frame graph giữ nguyên; codec vẫn
chạy fp32 với đúng các layer đó. Chỉ đổi *cách gọi* GPU cho cùng một phép tính: gom việc, bỏ đồng bộ thừa, bỏ copy thừa,
dùng CUDA graph. Mọi thay đổi phải qua bước kiểm tra tương đương trước khi đo tốc độ.

1. **Đo trước khi sửa.** Bật `PROFILE=1`, tạo tải, xem `/profile` (thời gian CPU/GPU theo từng vùng, TTFA tách theo
   từng chặng), mở trace trong Perfetto, đọc frame anatomy trong log. Kết luận: graph AR còn dư sức ~7 lần,
   còn codec bị giới hạn bởi CPU (182 ms CPU cho 54 ms GPU mỗi lần gọi).
2. **Đọc mã nguồn chỗ tốn nhất.** Tải source `vieneu==3.8.3` và code của codec trên HuggingFace về đọc. Tìm ra vòng lặp
   giải mã từng mẩu, các lệnh `.item()` theo từng slot, việc copy cả KV cache mỗi lần, và vì sao chế độ graph của
   upstream ra audio sai.
3. **Thiết kế thay thế giữ nguyên phép tính.** Transformer của codec là causal với context hữu hạn, nên giải mã
   N mẩu trong một lần forward cho đúng audio của N lần forward một mẩu; ring buffer giữ đúng tập key mỗi query nhìn
   thấy; CUDA graph chỉ ghi lại các kernel đó.
4. **Kiểm tra tương đương, trước khi đo tốc độ.**
   - Trên CPU với trọng số ngẫu nhiên: nhanh, không tốn tiền GPU, dùng để sửa logic ring.
   - Trên GPU với trọng số thật (`CODEC_VERIFY=1`, chạy mỗi lần khởi động): cùng chuỗi mã đi qua đường fp32 gốc của SDK
     và qua `SlotCodec`; stream dài 13 giây để ring quay vòng, có row vào giữa chừng, row rời đi, số mẩu lẫn lộn.
     Kết quả: correlation 0.9999993.
5. **Đo A/B trên cùng một máy.** Bản gốc (HEAD) được deploy từ một thư mục nháp lên chính máy đang thuê; benchmark chạy
   ngay trên máy GPU (`/benchmark`, 8 lượt mỗi mức) để tunnel không ảnh hưởng số đo.
6. **Đọc lại profile sau mỗi thay đổi** để tìm nút thắt kế tiếp: codec chuyển sang bị giới hạn bởi GPU; TTFA khi tải bị
   chi phối bởi hàng đợi và prefill.
7. **Kiểm tra ổn định.** Watchdog in stack nếu không có frame nào trong 30 giây; [stress.py](lite/stress.py) cho tải
   liên tục ở 32 client và 48 client (vượt số slot).
8. **Kiểm tra chất lượng.** CER bằng ASR khi tải so với khi chạy riêng; độ dài audio; clip im lặng hay bị cắt đỉnh; và
   [nghe so sánh](#nghe-so-sánh) bản gốc với bản tối ưu trên cùng câu, cùng seed.

Những cái bẫy đã gặp:

- **Tunnel đứt không phải GPU treo.** Một lần mình kết luận nhầm "overlap làm GPU treo ở 16 request"; thực ra tunnel VKong đứt.
  Chỉ tin vào stack dump của watchdog.
- **Hai CUDA graph chạy song song không được dùng chung capture stream** (chung workspace cuBLAS → GPU treo thật).
- **Không so dạng sóng ở chế độ greedy được** (model AR bf16 rẽ nhánh khi prefill gom batch khác nhau), nên chất lượng
  khi tải được đo bằng ASR.
- **Seed không tái lập được khi server đã chạy một lúc**: con trỏ ghi của KV cache AR dùng chung và tăng dần qua các
  request, nên hai lần đọc cùng câu, cùng seed trên một server ra hai chuỗi mã khác nhau (correlation 0.09–0.22).
  Chỉ ngay sau khi server khởi động, hai bản mới ở cùng một trạng thái.
- Máy thuê không cho khóa xung nhịp GPU, nên số đo dao động giữa các lần chạy.

### Profiler tìm ra gì

Đo bản lite cũ ở 1 request (RTX 3060, 16 slot, [results/profile-20261014](lite/results/profile-20261014/)):

| Phần | CPU | GPU | Nhận xét |
|---|---:|---:|---|
| Frame graph (một mẩu cho cả 16 slot) | 0.9 ms | 9–11.5 ms | Bản eager cần 1965 kernel, CPU mất 49–91 ms chỉ để launch; graph gần như không có khoảng hở (0.2 ms) |
| Một lần gọi codec (4 mẩu = 320 ms audio) | **182 ms** | ~54 ms | ~1750–2450 lần launch, ~3500 op, ~30% thời gian là Python giữa các op; riêng codec đã tốn RTF ~0.57 |
| Prefill người mới | ~35 ms | 4.6 ms | 738 lần launch (CPU đo khi không trace; trace làm chậm thêm, lên 46 ms) |

Graph AR chạy nhanh gấp ~7 lần real time; worker ngồi chờ codec (`decode.backpressure` ~129 ms mỗi lần gọi).
TTFA phía server 167 ms = admit (gồm prefill) 47 + lead-in 19 + lần gọi codec đầu 103.

Nguyên nhân nằm trong SDK của codec ([MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)):
`MossAudioTokenizerDecodeSession.step` giải mã **từng mẩu một** cho mọi slot và gọi `.item()` theo từng slot, mỗi mẩu
(CPU phải đợi GPU liên tục); attention streaming thì mỗi lần `torch.cat` cả KV cache, cắt, `where` rồi copy ngược
(~400 MB với 16 slot fp32, khoảng 5 lượt mỗi lần forward). Ghi chú của upstream nói chế độ `use_cuda_graph` của codec
"rẻ hơn 3 lần nhưng ra audio sai": `step` giữ các view trỏ vào buffer output tĩnh của graph qua từng mẩu, replay sau
ghi đè lên replay trước.

### Đã đổi gì

[codec.py](lite/vieneu_lite/codec.py) (`SlotCodec`) thay đường giải mã của SDK:

- **Slot của codec = slot của scheduler.** Job chạy theo thứ tự FIFO nên job kết thúc một row (và reset slot của nó)
  luôn chạy trước job đầu tiên của row kế tiếp vào slot đó; không còn ràng buộc "thứ tự trong list" và row giả của SDK.
- **Một lần forward cho cả job.** Các row có cùng số mẩu mới đi chung một `_decode_frame` (transformer của codec là
  causal với context hữu hạn, nhận chunk dài bao nhiêu cũng được); `exec_mask` giữ nguyên state của các slot khác.
- **CUDA graph** cho mỗi độ dài 1..4 mẩu trên tất cả slot, gồm cả việc chép `exec_mask`; output được chép ra tensor mới.
- **KV cache dạng ring** (`_ring_sdpa`): ring dài `context + 4 mẩu` mỗi layer, mỗi lần chỉ ghi vị trí mới.

[scheduler.py](lite/vieneu_lite/scheduler.py): ngoài mốc 4 mẩu chỉ giải mã những row mới tới lượt lead-in (mỗi lần forward
codec tốn như nhau, row cũ chờ tới mốc); mỗi tick nhận tối đa 16 request thay vì 8.

**Kiểm tra đúng:** `CODEC_VERIFY=1` cho cùng một chuỗi mã ngẫu nhiên đi qua đường gốc fp32 của SDK và qua `SlotCodec`
(2 row dài ~13 giây, vào giữa chừng, rời đi, số mẩu lẫn lộn, ring quay vòng): **correlation 0.9999993**, sai lệch lớn nhất
0.0105 trên thang ±1 (32 slot, graph + ring). Logic ring cũng được kiểm tra trên CPU với trọng số ngẫu nhiên.

**GPU treo và cách sửa:** `torch.cuda.graph` mặc định capture mọi graph trên cùng một stream; cuBLAS cấp workspace
theo stream, nên frame graph và codec graph cùng "đóng băng" một workspace. Khi hai graph chạy song song (overlap),
GPU treo lúc khởi động: watchdog cho thấy codec kẹt ở `.cpu()` còn worker kẹt trong `Event.record()`.
Capture codec graph trên stream riêng thì hết.

### Kết quả

**16 slot**, cùng một máy RTX 3060 (không khóa được xung nhịp GPU nên số dao động hơn bảng 13/10), client chạy trên
máy GPU, 8 lượt mỗi mức ([results/opt-20261014](lite/results/opt-20261014/)). Hai cột "mới" chưa có KV ring;
cột overlap bật đã có thay đổi lead-in/admit của scheduler.

| Request đồng thời | TTFA baseline (HEAD) | TTFA mới, overlap tắt | TTFA mới, overlap bật | RTF baseline | RTF mới, overlap tắt | RTF mới, overlap bật |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120.5 | **81.8** | 87.1 | 0.402 | 0.220 | **0.208** |
| 4 | 189.9 | **154.4** | 178.4 | 0.442 | 0.254 | **0.250** |
| 16 | 310.0 | **273.6** | 344.9 | 0.533 | 0.334 | **0.307** |

**32 slot, overlap bật, bản đầy đủ** (graph + ring), tải liên tục bằng [stress.py](lite/stress.py): mỗi client gửi
request liên tiếp, vào lệch giờ nhau để row liên tục vào/ra batch ([results/stress-20261014](lite/results/stress-20261014/)):

| Client đồng thời | Thời gian | Request | Lỗi | TTFA p50 / p99 | RTF p50 / p99 | Audio tạo ra mỗi giây |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3 phút | 1824 | 0 | 305 / 470 ms | 0.60 / 0.74 | 53 s |
| 48 (vượt 32 slot) | 2.5 phút | 1385 | 0 | 1833 / 2535 ms | 1.07 / 1.41 | 48 s |

Ở 48 client, request phải chờ slot trống nên TTFA và RTF tăng; không treo, không lỗi. Bảng 13/10 ghi 32 request
(32 slot) ở RTF 1.04, TTFA 591 ms, nhưng đó là máy khác và cách đo khác (dồn cùng lúc), chỉ để tham khảo.

### Nghe so sánh

Mỗi bản được khởi động lại rồi đọc lần lượt 5 câu dưới đây, từng request một, seed 42, 32 slot, tắt watermark.
Vì cả hai bản bắt đầu từ cùng một trạng thái, model AR sinh ra **cùng một chuỗi mã**; khác biệt chỉ còn đến từ codec.
Độ dài từng mẫu khớp tuyệt đối, correlation 0.9999998, sai lệch lớn nhất 0.0025 trên thang ±1
(SNR ~64 dB, tức phần khác nhau nhỏ hơn tín hiệu khoảng 1600 lần; tai người không phân biệt được).

| # | Câu | Giọng | Độ dài | Bản gốc (HEAD) | Bản tối ưu | SNR |
|---:|---|---|---:|---|---|---:|
| 1 | Xin chào, đây là giọng đọc VieNeu phát trực tiếp qua API chuẩn OpenAI. | Hải Đăng | 4.24 s | [original-1.wav](docs/samples/original-1.wav) | [optimized-1.wav](docs/samples/optimized-1.wav) | 64.2 dB |
| 2 | Hôm nay trời Hà Nội se lạnh, gió mùa đông bắc tràn về từ sáng sớm. | Hải Đăng | 3.68 s | [original-2.wav](docs/samples/original-2.wav) | [optimized-2.wav](docs/samples/optimized-2.wav) | 64.4 dB |
| 3 | VieNeu-TTS là mô hình đọc tiếng Việt chạy được thời gian thực. Gửi một đoạn văn, chỉ khoảng một phần mười giây sau đã nghe được tiếng đầu tiên. Phần còn lại được phát tiếp trong lúc mô hình vẫn đang tạo. | Hải Đăng | 9.12 s | [original-3.wav](docs/samples/original-3.wav) | [optimized-3.wav](docs/samples/optimized-3.wav) | 65.2 dB |
| 4 | Ngày 14 tháng 10 năm 2026, giá vàng tăng 1,5% lên 85 triệu đồng một lượng. | Hải Đăng | 5.68 s | [original-4.wav](docs/samples/original-4.wav) | [optimized-4.wav](docs/samples/optimized-4.wav) | 64.2 dB |
| 5 | Cảm ơn bạn đã lắng nghe, chúc bạn một ngày làm việc thật hiệu quả. | Trúc Ly | 3.04 s | [original-5.wav](docs/samples/original-5.wav) | [optimized-5.wav](docs/samples/optimized-5.wav) | 64.1 dB |

Số đo chi tiết: [docs/samples/compare.json](docs/samples/compare.json). Mẫu đọc khi server đang chịu tải 32 client
(chuỗi mã khác vì là các lần sinh khác nhau): [chạy riêng](lite/results/stress-20261014/wav/alone-0.wav) /
[khi tải](lite/results/stress-20261014/wav/load-0.wav) cho câu 1, và tương tự `alone-1/2`, `load-1/2` trong
[thư mục này](lite/results/stress-20261014/wav/).

### Chất lượng khi chạy batch lớn

`stress.py` lấy mẫu 150 audio trong lúc tải, cho [PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small) nhận
dạng sau khi tải xong, rồi tính CER (tỉ lệ sai ký tự) so với văn bản đầu vào; mỗi câu cũng được đọc 5 lần khi chạy một mình
để so. Nếu state giữa các slot bị lẫn hoặc reset sai, audio sẽ méo và CER khi tải cao hơn hẳn khi chạy riêng.

| | CER chạy riêng p50 / p90 | CER khi tải p50 / p90 | Clip > 20% |
|---|---:|---:|---:|
| 32 client | 4.4% / 11.8% | 1.8% / 13.2% | 4 / 150 |
| 48 client | 1.6% / 11.8% | 1.8% / 13.2% | 6 / 150 |

- Hai câu thuần Việt có CER trung vị 0% và 0.6% khi tải. Các clip CER cao đều là câu
  "…VieNeu phát trực tiếp qua API chuẩn OpenAI", ASR nghe sai phần tiếng Anh cả khi chạy riêng.
- Độ dài audio khi tải bằng 0.89–1.09 lần khi chạy riêng; không clip nào im lặng.
- 5–6% clip chạm ±1.0 và bị cắt đỉnh khi đổi sang PCM 16-bit. Codec mới cho dạng sóng gần như trùng codec cũ với cùng
  chuỗi mã, nên việc này đến từ mã do AR sinh ra; chưa đo tỉ lệ này trên bản cũ.
- Không so trực tiếp dạng sóng ở chế độ greedy được: model AR chạy bf16, prefill gom batch khác nhau (padding) và vị trí
  trong ring KV khác nhau làm `argmax` lật ở vài mẩu đầu, nên hai lần đọc cùng một câu ra hai chuỗi mã khác nhau.
- File nghe thử: [wav/](lite/results/stress-20261014/wav/) (chạy riêng, khi tải, và 3 clip CER tệ nhất).

### Rút ra

- Codec từng là phần đắt nhất. Giờ phía CPU của codec chỉ còn ~0.6 ms mỗi job (16 slot); RTF giảm khoảng một nửa,
  và 32 slot vẫn real time khi tải liên tục.
- Codec giờ bị giới hạn bởi GPU: ~28 ms mỗi lần forward với 16 slot, ~52 ms với 32 slot khi tải, vì luôn tính đủ mọi slot
  (kể cả slot trống), fp32.
- Overlap bật cho RTF tốt hơn, nhưng khi 16 request đến dồn cùng lúc thì TTFA tệ hơn cả baseline (345 so với 310 ms):
  job đầu của người mới phải chờ job codec đang chạy, cộng thời gian chờ admit.
- Việc còn lại cho TTFA: prefill vẫn eager (~35 ms CPU cho 4.6 ms GPU), có thể đưa vào CUDA graph theo nhóm độ dài;
  ưu tiên job lead-in của người mới; thử `CODEC_DTYPE=bf16` (cần kiểm tra lại chất lượng bằng ASR).
