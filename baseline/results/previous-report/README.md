# VieNeu SDK baseline

## Kết quả đo ngày 13/10/2026

RTX 3060 thuê qua VKong, Linux container, torch 2.8.0 + CUDA 12.8,
`vieneu==3.8.3`, transformers 4.57.6, sea-g2p 0.10.0. Giữ 16 slot,
`decode_frames=4`, `leadin_frames=2`, PCM 48 kHz, watermark tắt.
Client benchmark upstream chạy HTTP loopback trên GPU server, ba lượt mỗi mức tải:
**183/183 request thành công**. Khoảng TTFA dưới đây là min–max của ba median;
RTF max là giá trị lớn nhất trong ba lượt, tính riêng từng stream.

| Client đồng thời | TTFA median (ms) | TTFA max (ms) | RTF max |
|---:|---:|---:|---:|
| 1 | 130–135 | 135 | 0,56 |
| 4 | 208–222 | 225 | 0,67 |
| 8 | 200–240 | 248 | 0,64 |
| 16 | 250–264 | 428 | 0,72 |
| 32 | 1543–1615 | 3584 | 1,51 |

16 client vẫn sinh audio nhanh hơn thời gian thực trong bài test ngắn này.
32 client dùng chung 16 slot nên có xếp hàng, làm TTFA và RTF tăng rõ rệt.
Chưa đo p99, underrun, chất lượng nghe hoặc tải kéo dài; RTF tổng <1 chưa chứng
minh audio không bị ngắt giữa chừng. Dùng câu mặc định của client, giọng mặc định
SDK, sampling mặc định không cố định seed; chỉ warm-up khi startup, không khóa xung.

Raw log và runtime nằm trong [results/](results/); xem
[report.json](results/report.json) để lấy từng lượt và hash source.
Audio mẫu: [sample.wav](results/sample.wav). GPU đã dừng, không còn rental active.

## So với số liệu tác giả VieNeu

Đối chiếu gần nhất là bảng **Over HTTP** trong
[docs/streaming.md](https://github.com/pnnbao97/VieNeu-TTS/blob/bd0f88eb8de7d560753a6172ba9c868f309a101c/docs/streaming.md#over-http-appsopenai_speechpy-client-on-the-same-machine-watermark-on),
cùng RTX 3060, client cùng máy, 16 slot. Bảng này bật watermark; baseline tắt.

| Client | TTFA HTTP tác giả (ms) | TTFA baseline (ms) | Chênh TTFA | RTF tác giả | RTF max baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 106 | 130–135 | +23–27% | 0,47 | 0,56 |
| 4 | 160 | 208–222 | +30–39% | 0,51 | 0,67 |
| 8 | 207 | 200–240 | −3% đến +16% | 0,53 | 0,64 |
| 16 | 197 | 250–264 | +27–34% | 0,59 | 0,72 |

Bảng HTTP của tác giả ghi “RTF”, không nêu rõ thống kê max/median cho từng hàng;
cột baseline là **max**, nên không diễn giải chênh RTF thành phần trăm hồi quy.
Ở 16 client, TTFA max baseline 428 ms so với 336 ms tác giả công bố qua HTTP.

[README gốc](https://github.com/pnnbao97/VieNeu-TTS/blob/bd0f88eb8de7d560753a6172ba9c868f309a101c/README.md#benchmarks)
còn dẫn TTFA 115/164/185 ms và RTF 0,49/0,56/0,59 cho 1/8/16 stream;
đây là các số trong bảng **in-process**, không cùng đường HTTP của bài đo này.
RTF 0,011–0,02 trong phần throughput là batch synthesis, không phải RTF streaming
từng người nghe nên không dùng làm mốc so sánh ở đây.

Mốc **32 stream ~450–456 ms, RTF 0,93** của tác giả dùng `max_streams=32`.
Baseline vẫn là `max_streams=16` với 32 client chờ slot, nên **không so trực tiếp**
1543–1615 ms với mốc đó. Tác giả cũng ghi nhận request phải xếp hàng có TTFA 1,5–3 s
trong một kịch bản HTTP 40 client khác; đây chỉ là tham khảo về ảnh hưởng queueing.

Baseline hiện chậm hơn số công bố ở phần lớn mức tải, nhưng chưa phải phép đo A/B
cùng môi trường: tác giả dùng Windows 11 + i5 thế hệ 12, GPU warm; đây là Linux
container thuê, CPU/driver/xung khác và không đảm bảo GPU warm trước từng lượt.
SDK PyPI cũng không trùng hoàn toàn source `main`. Chưa profile từng thành phần
nên chưa kết luận nguyên nhân chênh lệch. Bản lite cần so với baseline đo ở đây,
với cùng tham số và benchmark, trước khi so tiếp với số tác giả.

## Lỗi gặp và cách xử lý

SDK 3.8.3 bị `IndexError: pop from empty list` ở scheduler admission khi gửi
32 request vào 16 slot mà không giới hạn HTTP: lượt đầu 32/32 lỗi.
Đã thêm semaphore 16 stream ở wrapper, tương tự giới hạn của API upstream,
rồi chạy lại toàn bộ bảng kết quả trên. Không sửa SDK hay benchmark.
Log trước sửa nằm trong [results/initial/](results/initial/).

## Chạy lại

- `serve.py`: FastAPI gọi SDK `Vieneu(...).infer_stream()`, PCM 48 kHz, 16 slot.
- `benchmark.py`: bản sao **nguyên file** [`examples/openai_speech_client.py`](https://github.com/pnnbao97/VieNeu-TTS/blob/bd0f88eb8de7d560753a6172ba9c868f309a101c/examples/openai_speech_client.py), commit `bd0f88eb8de7d560753a6172ba9c868f309a101c`.
- `vkong.yaml`: RTX 3060, cài `vieneu[cuda]==3.8.3` và chạy `serve.py`.

Chạy trong thư mục này:

```bash
vkong run
# Dùng port localhost mà vkong trả về; đổi concurrency thành 1/4/8/16/32:
curl -X POST 'http://127.0.0.1:<port>/benchmark?concurrency=8'
curl http://127.0.0.1:<port>/benchmark
curl http://127.0.0.1:<port>/results/bench-8.log
```

Endpoint chỉ gọi `python benchmark.py --bench N` trên GPU server rồi lưu stdout.
Chờ `running: false` trước case tiếp theo. Benchmark dùng HTTP loopback, không tính
WAN/tunnel. Mỗi lần chạy là một đợt N request, báo TTFA median/max và RTF max
của từng stream. Code gốc không đo p99 hoặc underrun. c=32 có queueing vì server
chỉ có 16 slot (semaphore ở HTTP giới hạn số infer_stream đang chạy). Server warm-up khi khởi động; watermark tắt.

Nếu đã vào shell của máy GPU, chạy trực tiếp trong thư mục chứa `serve.py`:

```bash
python benchmark.py --base http://127.0.0.1:8000 --bench 8
```

Các số trong report đều đo bằng subprocess trên máy GPU. Lệnh `curl /benchmark`
từ máy cá nhân chỉ khởi chạy job và lấy log; request sinh audio và đồng hồ đo
TTFA/RTF đều nằm trên máy GPU.

Serving này hỗ trợ PCM 48 kHz; các tùy chọn SSE/resample của client gốc không được
hỗ trợ. Tải log về `results/` trước khi stop. Sửa trên rental hiện tại:

```bash
vkong attach <instance-id> --update  # code
vkong attach <instance-id> --setup   # dependencies
vkong app stop vieneu-original-baseline
```

Cài SDK theo [upstream](https://github.com/pnnbao97/VieNeu-TTS/tree/main#installation).
Workflow theo [vkong-cli skill](https://vkong.tli-tech.com/download/skills/vkong-cli)
revision `2026-10-11.5`. Setup/download cũng tính tiền; `max_dph` là trần giá/giờ.

Kết quả đã đo: [results/report.md](results/report.md), kèm raw log và runtime.
