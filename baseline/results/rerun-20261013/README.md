# VieNeu SDK baseline

## Kết quả đo lại — 13/10/2026

**Cùng máy GPU, cùng container; benchmark gọi localhost ngay trên máy server.**
Baseline và lite mỗi bản **183/183 request thành công**. Đã tắt rental sau đo.

| Clients | Baseline TTFA (ms) | Lite TTFA (ms) | Chênh TTFA lite | Baseline RTF max | Lite RTF max |
|---:|---:|---:|---:|---:|---:|
| 1 | 132–135 | 132–149 | +6.7% | 0.55 | 0.56 |
| 4 | 160–187 | 186–209 | +10.1% | 0.59 | 0.64 |
| 8 | 208–236 | 223–243 | +8.4% | 0.63 | 0.67 |
| 16 | 248–266 | 264–282 | +1.9% | 0.71 | 0.71 |
| 32 | 1460–1706 | 1682–1697 | +0.7% | 1.48 | 1.57 |

Trong lượt đo lại, TTFA lite cao hơn baseline **0.7–10.1%** theo median của 3 median.
Ở 16 clients, chênh TTFA là **1.9%**, RTF max cùng **0.71**; cả hai sinh nhanh hơn
thời gian thực trong bài test ngắn này. Ở 32 clients, RTF max cả hai vượt 1 do chờ slot.

TTFA: khoảng median của 3 lượt; chênh lệch tính từ median của 3 median; RTF là max.
Cùng 16 slots, K=4, leadin=2, watermark tắt. Xem [report đầy đủ](results/report.md)
và [số liệu từng lượt](results/report.json). Không đảo thứ tự chạy hay khóa GPU clock;
chưa đo p99, underrun và tải dài hạn. Snapshot GPU có tải trước load lite, nên chưa
kết luận chênh lệch do riêng code. Số cũ nằm trong `results/previous-report/`.

## So với số tác giả VieNeu

Mốc gần nhất là bảng [Over HTTP](https://github.com/pnnbao97/VieNeu-TTS/blob/bd0f88eb8de7d560753a6172ba9c868f309a101c/docs/streaming.md#over-http-appsopenai_speechpy-client-on-the-same-machine-watermark-on),
cũng RTX 3060, client cùng máy, 16 slot nhưng **bật watermark**; bài đo này tắt.

| Clients | TTFA tác giả (ms) | TTFA baseline mới (ms) | RTF tác giả | RTF max baseline |
|---:|---:|---:|---:|---:|
| 1 | 106 | 132–135 | 0.47 | 0.55 |
| 4 | 160 | 160–187 | 0.51 | 0.59 |
| 8 | 207 | 208–236 | 0.53 | 0.63 |
| 16 | 197 | 248–266 | 0.59 | 0.71 |

Tác giả không nêu rõ RTF của bảng HTTP là max hay median, nên không tính phần trăm
chênh RTF. Môi trường tác giả là Windows 11, i5 thế hệ 12, GPU warm; đây là Linux thuê.
Không thể quy chênh lệch cho code. Mốc 32 streams ~456 ms/RTF 0.93 dùng 32 slots,
khác 32 clients chờ 16 slots ở đây. Số in-process và RTF batch 0.011–0.02 trong
[README upstream](https://github.com/pnnbao97/VieNeu-TTS/blob/bd0f88eb8de7d560753a6172ba9c868f309a101c/README.md#benchmarks)
không dùng làm mốc HTTP streaming.

## Lỗi SDK đã xử lý trước benchmark

SDK 3.8.3 từng lỗi `free.pop(0)` khi 32 requests vào 16 slots. Wrapper dùng
`BoundedSemaphore(16)`; không sửa SDK. Log cũ: [results/initial/](results/initial/).

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

Endpoint gọi `python benchmark.py --base http://127.0.0.1:8000 --bench N` trên GPU server rồi lưu stdout.
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
