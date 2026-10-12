# VieNeu SDK 3.8.3 — baseline RTX 3060

Đo lúc 2026-10-12T17:51:02.341367+00:00. HTTP loopback trên GPU, ba lượt mỗi mức tải, **183/183 request thành công**.

16 slot, K=4, leadin=2, PCM 48 kHz; watermark tắt. SDK và benchmark upstream không sửa.

| Client đồng thời | TTFA median qua 3 lượt (ms) | TTFA max (ms) | RTF max |
|---:|---:|---:|---:|
| 1 | 130–135 | 135 | 0.56 |
| 4 | 208–222 | 225 | 0.67 |
| 8 | 200–240 | 248 | 0.64 |
| 16 | 250–264 | 428 | 0.72 |
| 32 | 1543–1615 | 3584 | 1.51 |

Ở 16 client, RTF max 0,72: tốc độ sinh audio còn nhanh hơn thời gian thực trong bài test ngắn này. Ở 32 client, vẫn chỉ có 16 slot; queueing đẩy TTFA median lên 1,54–1,62 giây và RTF max lên 1,51.

TTFA là median/max do client gốc in ra, RTF max tính trên từng stream. Khoảng trong bảng là min–max của ba median, không phải percentile gộp. Không đo p99, underrun, chất lượng nghe hoặc tải kéo dài. Không khóa xung GPU, không đặt seed sampling. Câu test và giọng mặc định giống client upstream.

Lần đầu ở c=32 chưa có giới hạn HTTP: 32/32 request lỗi do SDK `free.pop(0)` khi nhận quá số slot trống. Đã thêm semaphore 16 stream ở wrapper, như giới hạn admission của API upstream, rồi chạy lại toàn bộ bảng trên. SDK không bị patch. Log thử trước nằm trong `initial/`.

Raw log: `bench-c*-r*.log`; runtime và số liệu: `runtime.json`, `report.json`; audio mẫu: `sample.wav`.

Triển khai và sửa bằng `vkong run` / `vkong attach --update` trên cùng rental `vk_8288991c1e0dc349`. Giá $0,064445/giờ. Lỗi start với biến môi trường inline đã gửi qua skill vkong-cli, feedback `fb-128`; workaround đặt biến trong Python.

Cleanup: App đã stopped, rental không còn trong danh sách active; không có Storage được tạo.
