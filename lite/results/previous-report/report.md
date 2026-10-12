# Kết quả deploy và benchmark lite — 13/10/2026

**183/183 request thành công.** Lite chậm hơn baseline trong các lượt đo này. Hai lượt deploy dùng **máy thuê khác nhau**, cùng RTX 3060 12 GB; chưa thể quy chênh lệch cho riêng code.

| Clients | Baseline TTFA (ms) | Lite TTFA (ms) | Chênh TTFA | Baseline RTF max | Lite RTF max |
|---:|---:|---:|---:|---:|---:|
| 1 | 130–135 | 184–186 | +39% | 0.56 | 0.79 |
| 4 | 208–222 | 254–262 | +25% | 0.67 | 0.88 |
| 8 | 200–240 | 303–329 | +36% | 0.64 | 0.93 |
| 16 | 250–264 | 333–358 | +35% | 0.72 | 1.03 |
| 32 | 1543–1615 | 2174–2400 | +54% | 1.51 | 2.09 |

TTFA: khoảng median của 3 lượt; chênh lệch: tính từ median của 3 median; RTF: max của cả 3 lượt, càng thấp càng tốt. Ở 16 clients, lite đạt RTF max 1.01–1.03, baseline 0.69–0.72. Cả hai vượt RTF 1 ở 32 clients do chỉ có 16 slot và phải chờ. TTFA lớn nhất ở 16/32 clients: lite 578/4888 ms, baseline 428/3584 ms.

## Cách đo

- `benchmark.py` giữ nguyên từ upstream `examples/openai_speech_client.py`, commit `bd0f88eb8de7d560753a6172ba9c868f309a101c`. SHA-256: `f2e1edf08fc3c2fea9c203c7ed4416a881d55d359bec751e005866872782abfb`.
- Client chạy HTTP loopback trên GPU server, không tính tunnel/WAN. 3 lượt mỗi mức, thứ tự 32, 1, 4, 8, 16; mỗi lượt là burst ngắn.
- Cùng VieNeu 3.8.3, torch 2.8.0+cu128, transformers 4.57.6; 16 slots, decode K=4, leadin=2, watermark tắt, PCM 48 kHz.
- Cùng câu mặc định: “Xin chào, đây là giọng đọc VieNeu phát trực tiếp qua API chuẩn OpenAI.” `voice=None` dùng preset mặc định SDK; warmup trước đo, không cố định seed hay GPU clock.
- Baseline dùng semaphore giới hạn 16 lời gọi SDK; lite xếp hàng trong scheduler. 32 clients không phải 32 slot hoạt động.
- Chưa đo p99, underrun, tải dài hạn hay chất lượng giọng. WAV lite kiểm tra cấu trúc hợp lệ, dài 4.16 s; chưa đánh giá nghe.

## Deploy và xử lý lỗi

Đổi folder `serving-lite` thành `lite`. App `vieneu-serving-lite`, giá thuê $0.064445/giờ. Sửa lệnh khởi động thành `env WATERMARK=0 OMP_NUM_THREADS=4 python -u serve.py` và in `nvidia-smi` trước load; không sửa lõi lite trong lượt deploy này.

Máy đầu đã dùng 12029/12288 MiB VRAM trước load model; dùng attach để chẩn đoán rồi thay rental. Máy đo chính `vk_d3719a010b81d1e7`, driver 535.183.01, trước load chỉ dùng 1 MiB. Xem `startup-failure.txt` và `gpu-startup.txt`. Đã gửi phản hồi VKong `fb-129` theo [skill VKong](https://vkong.tli-tech.com/download/skills/vkong-cli).

Đã attach baseline lên máy này để chuẩn bị đối chứng nhưng dừng theo yêu cầu người dùng, **không chạy benchmark đối chứng cùng máy**. Bảng trên dùng baseline đã lưu từ rental `vk_8288991c1e0dc349`. Instance đã được tắt sau đo.

Chi tiết từng lượt và hash source: [report.json](report.json). Log gốc: `bench-cN-rR.log`. So sánh baseline với số tác giả công bố: [baseline README](../../../baseline/results/previous-report/README.md).
