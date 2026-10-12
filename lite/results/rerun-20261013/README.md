# VieNeu serving lite

Bản serving gọn (chỉ GPU, chỉ streaming) theo [mô tả cách hoạt động trong README gốc](../../../README.md#vieneu-hoạt-động-như-thế-nào).
Mới viết sạch lại phần lõi, **chưa tối ưu** (chưa làm mục §7).

- `vieneu_lite/frame.py`: frame CUDA graph (`RepHistory`, `sample_rows`, `StaticBackbone`, `StreamFrame`), §3.
- `vieneu_lite/scheduler.py`: continuous batching, `codec_rows`/finalize, leadin, hủy, cô lập lỗi, §4–5.
- `vieneu_lite/tts.py`: cắt chunk text, nối chunk qua `gen_done`, pause padding, watermark, §4.5.
- `serve.py`: `POST /v1/audio/speech` (pcm/wav 48 kHz), `GET /v1/voices`, `GET /health` (503 khi worker chết),
  cùng `/benchmark` như baseline.
- `benchmark.py`, `requirements.txt`, `.vkongignore`, `.gitignore`: dùng cùng cấu hình với `../baseline`.

Package `vieneu==3.8.3` vẫn lo việc load model, codec, tokenizer, giọng preset, phonemizer, watermark
và các tiện ích chunk/pause.

Biến môi trường: `MAX_STREAMS` (16), `DECODE_FRAMES` (4), `LEADIN_FRAMES` (2), `WATERMARK` (1), `PORT` (8000).
`vkong.yaml` đặt `WATERMARK=0` để so sánh công bằng với baseline.

Chạy và benchmark giống baseline:

```bash
vkong run
curl -X POST 'http://127.0.0.1:<port>/benchmark?concurrency=8'
curl http://127.0.0.1:<port>/benchmark
curl http://127.0.0.1:<port>/results/bench-8.log
```

`curl /benchmark` chỉ khởi chạy job trên máy GPU. `serve.py` tạo subprocess
`python benchmark.py --bench N` ngay trong cùng container; client gọi
`http://127.0.0.1:8000`. TTFA/RTF được đo tại đó, không đi qua tunnel.
Toàn bộ số trong report dùng cách này.

Nếu đã vào shell máy GPU, chạy trực tiếp trong thư mục chứa `serve.py`:

```bash
python benchmark.py --base http://127.0.0.1:8000 --bench 8
```

Khác biệt so với `stream.py` gốc (để giữ code gọn, hành vi không đổi):

- `recent_codes` giữ code trên GPU, không đi vòng GPU→CPU→GPU. Ring code dài 64 frame thay vì 512.
- Prompt 2D được dựng ở thread của caller. Ở đó cũng kiểm tra sampling và độ dài prompt.
- Embed/giọng lỗi chỉ làm fail request đó. OOM khi prefill chỉ làm fail các request trong lần prefill đó.

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

So sánh baseline với upstream: [baseline README](../../../baseline/results/previous-report/README.md).
