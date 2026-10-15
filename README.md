# VieNeu-TTS inference

## Vì sao làm

[VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) v3 Turbo là model đọc tiếng Việt chạy được thời gian thực:
gửi một đoạn văn, khoảng 130 ms sau đã nghe được tiếng đầu tiên, phần còn lại được stream tiếp trong lúc model
vẫn đang tạo. Tốc độ đó không nằm ở bản thân model (khá nhỏ), mà ở cách server chạy nó: CUDA graph,
nhiều người dùng chung một lần chạy GPU, và codec giải mã từng đoạn ngắn.

Mục tiêu của repo: hiểu cách server đó chạy, tự chạy nó trên GPU thuê qua [VKong](https://vkong.tli-tech.com) do AI
agent điều khiển (deploy, sửa lỗi, benchmark, tải log, trả máy), rồi làm nó nhanh hơn mà không đụng vào model.

## Đạt được gì

Bản lite sau khi tối ưu codec, đo trên cùng một máy RTX 3060 với 16 ghế (slot), so với bản trước đó:

| | RTF 1 người | RTF 16 người | Tiếng đầu (TTFA), 1 người | TTFA, 16 người vào cùng lúc |
|---|---:|---:|---:|---:|
| Trước | 0.402 | 0.538 | 124 ms | 294 ms |
| Sau | **0.175** | **0.250** | **76 ms** | 300 ms |

**RTF** là thời gian làm chia cho độ dài âm thanh; dưới 1 là người nghe không bị ngắt. Nhanh hơn gấp đôi khi ít người
dùng; riêng tiếng đầu lúc nhiều người vào cùng lúc thì chưa cải thiện.

- **Âm thanh không đổi.** Giải mã cùng một chuỗi mã thì tương quan 0.9999993. Mẫu nghe cùng câu, cùng seed:
  [trước](docs/samples/original-1.wav) và [sau](docs/samples/optimized-1.wav), đủ 5 câu trong [docs/samples/](docs/samples/).
- **Ổn định khi tải lớn.** 32 ghế, 32 người dùng liên tục trong 3 phút: 1824 request, không lỗi; máy nhận dạng giọng
  nói nghe đúng như lúc chạy riêng.
- **Không đụng vào model.** Trọng số, kiến trúc, kiểu số và cách sampling giữ nguyên; chỉ đổi cách codec chạy trên
  GPU, trong [codec.py](lite/vieneu_lite/codec.py).

[![Video: tối ưu VieNeu lite](docs/optimize.png)](docs/optimize.mp4)

*Video 3 phút 10 giây về phần tối ưu, lồng tiếng bằng chính bản tối ưu; phụ đề:
[optimize.vi.srt](docs/optimize.vi.srt). Câu chuyện đầy đủ (đo gì, đoán ra điều gì, sửa ra sao, kiểm tra thế nào trên
VKong ở từng bước, các lần đi sai đường): **[OPTIMIZATION.md](OPTIMIZATION.md)**.*

## Ban đầu tiếp cận khác

Lần đầu (13/10), mình tiếp cận từ phía code: đọc hiểu đường inference của upstream, **viết lại một bản đơn giản**
(lite) chỉ giữ đường streaming trên GPU, đủ nhỏ để đọc hết trong một buổi, rồi thử đổi cách chạy (cho codec chạy song
song với model). Kết quả: lite chạy **xấp xỉ SDK gốc, không nhanh hơn**
([bảng A–B–B–A](#kết-quả-lần-đầu-lite-so-với-sdk-gốc-rtx-3060-13102026)), và chạy song song chỉ nhích một chút. Phần làm
nên tốc độ là thiết kế của tác giả VieNeu; viết lại chỉ giúp code dễ đọc.

Lần sau (14–15/10), mình đổi cách: **không đoán, đo trước**. Gắn profiler, đo trên GPU thật và thấy model sinh âm đã
nhanh (còn dư sức ~7 lần), còn chỗ chậm là codec: mỗi lần gọi, CPU bận 182 ms để ra lệnh trong khi GPU chỉ tính ~54 ms.
Đọc code codec ra ba nguyên nhân và sửa đúng ba chỗ đó: **gom 4 mẩu vào một lượt**, **CUDA graph**, và **KV cache dạng
vòng**. Chi tiết trong [OPTIMIZATION.md](OPTIMIZATION.md).

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
| [codec.py](lite/vieneu_lite/codec.py) | Codec MOSS chạy streaming theo slot: một lần forward cho cả job, CUDA graph, KV cache dạng ring (xem [OPTIMIZATION.md](OPTIMIZATION.md)) |
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

Khi dùng VKong để đo và sửa (tunnel hay đứt, volume chặn `attach --update`, chốt giá trước khi thuê…), xem phần
cuối của [OPTIMIZATION.md](OPTIMIZATION.md).

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

## Kết quả lần đầu: lite so với SDK gốc (RTX 3060, 13/10/2026)

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
