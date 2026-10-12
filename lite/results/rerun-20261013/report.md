# Benchmark lại baseline và lite trên cùng máy

Đo lại ngày **13/10/2026**, lần lượt baseline rồi lite trên **cùng rental và container**:
`vk_8cdb400eca7bc18b`, hostname `c5553ff94ac0`. Mỗi bản **183/183 request thành công**,
tổng **366/366**, không lỗi. Bảng này thay thế so sánh hai máy ở lượt trước.

| Clients | Baseline TTFA (ms) | Lite TTFA (ms) | Chênh TTFA lite | Baseline RTF max | Lite RTF max |
|---:|---:|---:|---:|---:|---:|
| 1 | 132–135 | 132–149 | +6.7% | 0.55 | 0.56 |
| 4 | 160–187 | 186–209 | +10.1% | 0.59 | 0.64 |
| 8 | 208–236 | 223–243 | +8.4% | 0.63 | 0.67 |
| 16 | 248–266 | 264–282 | +1.9% | 0.71 | 0.71 |
| 32 | 1460–1706 | 1682–1697 | +0.7% | 1.48 | 1.57 |

TTFA là khoảng min–max của 3 median; chênh lệch dùng median của 3 median,
không phải median gộp mọi request. RTF max là request có RTF lớn nhất trong 3 lượt,
càng thấp càng tốt. Xem [report.json](report.json) để lấy từng lượt.

Trong lượt đo lại, TTFA lite cao hơn baseline **0.7–10.1%** theo median của 3 median.
Ở 16 clients, chênh TTFA là **1.9%**, RTF max cùng **0.71**; cả hai sinh nhanh hơn
thời gian thực trong bài test ngắn này. Ở 32 clients, RTF max cả hai vượt 1 do chờ slot.

## Phép đo

- `benchmark.py` nguyên bản upstream `examples/openai_speech_client.py`, commit `bd0f88eb8de7d560753a6172ba9c868f309a101c`, SHA-256 `f2e1edf08fc3c2fea9c203c7ed4416a881d55d359bec751e005866872782abfb`.
- Server tạo subprocess ngay trong container, chạy `python benchmark.py --base http://127.0.0.1:8000 --bench N`. Mọi request audio và đồng hồ TTFA/RTF ở trên máy GPU. Máy cá nhân chỉ kích hoạt job và tải log; thời gian tunnel không nằm trong số đo.
- Log mới có `benchmark_host=c5553ff94ac0`, baseline `server_pid=202`, lite `server_pid=861`, cùng `base=http://127.0.0.1:8000`.
- Mỗi bản chạy 3 lượt tại mỗi mức, thứ tự 32, 1, 4, 8, 16; warmup lúc startup. Baseline chạy trước, sau đó `vkong attach --update` đổi sang lite; không chạy đồng thời hai model và không cài lại dependencies.
- Cùng RTX 3060 12 GB, CPU được quote 10.7, RAM 21 GB, disk 40 GB, Quebec. Driver 560.35.03, giới hạn công suất 130 W theo snapshot startup lite. Giá $0.064445/giờ.
- Cùng VieNeu 3.8.3, torch 2.8.0+cu128, transformers 4.57.6, sea-g2p 0.10.0; 16 slots, K=4, leadin=2, watermark tắt, PCM 48 kHz.
- Câu mặc định: “Xin chào, đây là giọng đọc VieNeu phát trực tiếp qua API chuẩn OpenAI.” `voice=None` dùng Hải Đăng; sampling mặc định, không cố định seed. Baseline giới hạn 16 lời gọi SDK bằng semaphore; lite xếp hàng trong scheduler. 32 clients vẫn chỉ dùng 16 slot.

## Giới hạn

Đây là burst ngắn, chưa phải kiểm thử tải kéo dài. Không đo p99, underrun hay chất lượng nghe.
Không đảo thứ tự baseline/lite, không khóa GPU clock. Snapshot trước load lite ghi GPU-Util 99%
với 129 MiB VRAM; chưa xác định nguyên nhân và chưa chứng minh GPU được dùng độc quyền.
Vì vậy kết quả mô tả hiệu năng quan sát được trên cùng rental, chưa xác định nguyên nhân chênh lệch.

## Thay đổi và bằng chứng

Chỉ sửa wrapper benchmark để truyền rõ loopback và ghi hostname/PID; giữ nguyên client upstream
và lõi inference. Lỗi dấu nháy f-string của wrapper lite trên Python 3.11 đã sửa bằng
`attach --update` trước khi chạy benchmark lite; các lượt lỗi khởi động không được tính vào bảng.
Lite được stage với app name `vieneu-original-baseline` để tái sử dụng rental, các file Python
khớp folder `lite/`. App name trong config gốc của lite vẫn là `vieneu-serving-lite`.

Log và runtime mới: [rerun-20261013/](rerun-20261013/). Hash source: [source-sha256.json](rerun-20261013/source-sha256.json).
Report cũ được giữ trong `previous-report/`; log ngoài thư mục rerun là dữ liệu lượt trước.
Đã xuất kết quả rồi tắt rental; trạng thái sau tắt: [vkong-after-stop.json](rerun-20261013/vkong-after-stop.json).

Snapshot GPU: [startup.log](rerun-20261013/startup.log).
