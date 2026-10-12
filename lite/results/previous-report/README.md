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

## Kết quả trên VKong — 13/10/2026

**183/183 request thành công.** Lite chậm hơn baseline trong các lượt đo này. Hai lượt deploy dùng **máy thuê khác nhau**, cùng RTX 3060 12 GB; chưa thể quy chênh lệch cho riêng code.

| Clients | Baseline TTFA (ms) | Lite TTFA (ms) | Chênh TTFA | Baseline RTF max | Lite RTF max |
|---:|---:|---:|---:|---:|---:|
| 1 | 130–135 | 184–186 | +39% | 0.56 | 0.79 |
| 4 | 208–222 | 254–262 | +25% | 0.67 | 0.88 |
| 8 | 200–240 | 303–329 | +36% | 0.64 | 0.93 |
| 16 | 250–264 | 333–358 | +35% | 0.72 | 1.03 |
| 32 | 1543–1615 | 2174–2400 | +54% | 1.51 | 2.09 |

TTFA: khoảng median của 3 lượt; chênh lệch: tính từ median của 3 median; RTF: max của cả 3 lượt, càng thấp càng tốt. Ở 16 clients, lite đạt RTF max 1.01–1.03, baseline 0.69–0.72. Cả hai vượt RTF 1 ở 32 clients do chỉ có 16 slot và phải chờ. TTFA lớn nhất ở 16/32 clients: lite 578/4888 ms, baseline 428/3584 ms.

Cùng 16 slots, K=4, leadin=2, watermark tắt, benchmark upstream nguyên bản. Instance đã tắt. Xem [report đầy đủ](results/report.md), [số liệu từng lượt](results/report.json) và [baseline so với upstream](../../../baseline/results/previous-report/README.md). Chưa đo p99, underrun hay tải dài hạn.
