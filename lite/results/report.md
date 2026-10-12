# Benchmark A–B–B–A — 13/10/2026

**1.260/1.260 request đo chính thành công**, mỗi bản 630, trên cùng rental
`vk_7fba8887e3fa2c37`. Chạy lần lượt **baseline → lite → lite → baseline**;
tắt process cũ trước khi start process mới. Mỗi block đo 16 slots ở c=1/4/8/16,
sau đó restart với 32 slots để đo c=32. Không chạy hai model đồng thời.

## Kết quả theo từng request

| Clients | Slots | n mỗi bản | Baseline TTFA p50/p90 (ms) | Lite TTFA p50/p90 (ms) | Baseline RTF p50/p90 | Lite RTF p50/p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 30 | 127.0 / 132.6 | 129.1 / 136.9 | 0.529 / 0.550 | 0.548 / 0.567 |
| 4 | 16 | 40 | 174.4 / 186.5 | 173.6 / 191.1 | 0.568 / 0.599 | 0.568 / 0.620 |
| 8 | 16 | 80 | 202.6 / 222.3 | 202.5 / 221.3 | 0.591 / 0.642 | 0.590 / 0.643 |
| 16 | 16 | 160 | 243.6 / 404.6 | 273.8 / 428.1 | 0.674 / 0.701 | 0.659 / 0.725 |
| 32 | 32 | 320 | 588.4 / 798.2 | 591.5 / 850.1 | 1.040 / 1.125 | 1.039 / 1.090 |

p50/p90 tính trực tiếp trên từng request thành công, nội suy tuyến tính tại `(n−1)×p`.
Không lấy median của các median. Mỗi bản có 30 request tuần tự ở c=1; các mức khác
có 10 burst (5 burst mỗi block). Mỗi case có 2 burst warmup, loại khỏi thống kê.
RTF là thời gian client từ trước POST đến hết audio chia cho độ dài audio.

## Nhận xét từ lần đo này

- c=1/4/8/32 có TTFA p50 gộp gần nhau; không thấy lite chậm hơn đồng loạt.
- c=16: p50 gộp baseline **243.6 ms**, lite **273.8 ms**; p90 **404.6/428.1 ms**. Tuy nhiên lite B1/B2 là **229.9/277.4 ms**, còn baseline A1/A2 **254.7/235.7 ms**. Một block lite thấp hơn cả hai block baseline, block còn lại cao hơn: chưa có hiệu ứng chậm hơn ổn định qua các block.
- Chênh c=16 cũng xuất hiện ở server TTFA (p50 **223.3/255.8 ms**), nên không thể giải thích chỉ bằng tunnel; tunnel vốn không nằm trong phép đo. Chưa profile để xác định nguyên nhân.
- c=32/32 slots: TTFA p50 **588.4/591.5 ms**, RTF p50 **1.040/1.039**. RTF client vẫn gồm thời gian chuẩn bị/admission và nhận audio; không đồng nghĩa với phép đo tốc độ decode thuần hay bằng chứng underrun.
- Bộ đo này tốt hơn lượt ba mẫu vì có A–B–B–A và raw request, nhưng **chưa đủ để tuyên bố tương đương hoặc quy chênh lệch cho code**: chưa khóa được xung và chỉ có hai block mỗi bản.

## TTFA phía server

| Clients | Baseline server TTFA p50/p90 (ms) | Lite server TTFA p50/p90 (ms) |
|---:|---:|---:|
| 1 | 123.6 / 128.7 | 126.5 / 133.5 |
| 4 | 165.1 / 178.9 | 164.6 / 184.5 |
| 8 | 189.4 / 206.7 | 190.8 / 205.5 |
| 16 | 223.3 / 376.3 | 255.8 / 403.4 |
| 32 | 536.2 / 740.9 | 557.9 / 811.9 |

Ghép đủ 1.260 cặp client/server bằng request ID. Server TTFA đo từ handler sau
validation đến chunk float đầu tiên, gồm chuẩn bị và scheduler; chưa gồm chuyển PCM,
ASGI/socket. Client TTFA còn gồm HTTP và nhận byte đầu. Hai mốc không hoàn toàn giống nhau.

## Kiểm tra thứ tự chạy

TTFA p50 theo từng block, đơn vị ms:

| Clients | A1 baseline p50 | B1 lite p50 | B2 lite p50 | A2 baseline p50 |
|---:|---:|---:|---:|---:|
| 1 | 129.7 | 128.9 | 130.3 | 120.3 |
| 4 | 172.2 | 164.5 | 186.6 | 178.3 |
| 8 | 202.2 | 194.7 | 216.8 | 203.4 |
| 16 | 254.7 | 229.9 | 277.4 | 235.7 |
| 32 | 652.3 | 583.4 | 653.0 | 578.2 |

A–B–B–A giúp nhìn được biến động theo lượt; chỉ có hai block mỗi bản, chưa phải kiểm
định tương đương hay bằng chứng nhân quả về từng tối ưu trong code.

## Điều kiện đo và kiểm chứng

- Benchmark process chạy **trong cùng container với server**, gọi rõ `http://127.0.0.1:8000`. Máy cá nhân chỉ kích hoạt job và tải artifacts, không đo TTFA qua tunnel.
- Cùng RTX 3060 12 GB, CPU quote 9, RAM 31.5 GB, disk 40 GB, Việt Nam. Giá $0.077778/giờ. Driver 595.104.02, giới hạn công suất **120 W**; không sửa power limit.
- VieNeu 3.8.3, torch 2.8.0+cu128, transformers 4.57.6, sea-g2p 0.10.0; K=4, leadin=2, watermark tắt, PCM 48 kHz, cùng câu mặc định và giọng Hải Đăng.
- Trước **mỗi lần load model** và **mỗi case concurrency**, đợi GPU-Util ≤5% trong 5 mẫu liên tiếp. Cả 8 lần load và 20 case đều qua kiểm tra. Snapshot và telemetry trong lúc đo được lưu riêng từng case.
- Đã thử `nvidia-smi -i 0 -lgc 1500,1500`; driver trả **exit 4: không có quyền đổi xung**. **Không khóa được xung**. Không tăng quyền hay can thiệp host. Xung/nhiệt độ dưới đây là số quan sát, không phải cấu hình cố định.
- Đặt lại seed Python/NumPy/Torch trước từng burst: `42 + chỉ số lượt`, cùng lịch seed giữa các block. Không reset seed giữa các request đang chạy. Scheduling và CUDA graph vẫn có thể làm audio khác nhau; đây không phải đảm bảo deterministic tuyệt đối.
- Hash đọc trực tiếp từ file **trên máy GPU** ở mỗi case khớp hash local cho `serve.py`, `benchmark.py`, `bench_support.py` và các file lõi lite. Cùng hostname và GPU UUID trong mọi block.

- baseline: xung khi GPU-Util ≥50% 1777–1935 MHz; nhiệt độ trong các mẫu telemetry 54–71 °C.
- lite: xung khi GPU-Util ≥50% 1792–1935 MHz; nhiệt độ trong các mẫu telemetry 53–71 °C.

Độ dài audio đo được (sampling vẫn có biến động):

| Clients | Baseline audio min–max (s) | Lite audio min–max (s) |
|---:|---:|---:|
| 1 | 3.84–4.24 | 3.84–4.24 |
| 4 | 3.60–4.40 | 3.60–4.40 |
| 8 | 3.68–4.56 | 3.68–4.40 |
| 16 | 3.52–4.56 | 3.52–4.48 |
| 32 | 3.68–4.80 | 3.68–4.80 |


## Phạm vi so sánh

**Baseline là SDK nguyên bản với wrapper HTTP có `BoundedSemaphore`, không phải
server upstream nguyên bản.** Lõi SDK giữ nguyên; semaphore tránh lỗi admission khi
overload. Lite giữ nguyên lõi inference trong lượt đo này. Cả hai thêm cùng cơ chế
trace và reset seed dành cho benchmark, bật bằng `BENCH_MODE=1`.

`benchmark.py` dựa trên client upstream và giữ đường PCM/cách bấm thời gian của
`stream_pcm`; có mở rộng request ID, barrier, lặp lượt, seed và JSONL. Vì vậy bản đo
mới **không còn là bản sao nguyên file upstream**. Bản upstream gốc được lưu trong
`abba-20261013/benchmark-upstream.py` để đối chiếu.

Ở c=32, **MAX_STREAMS=32**, không trộn nhóm chờ giải phóng 16 slots như report cũ.
Scheduler vẫn có admission/prefill và batching, nên không gọi đây là đo kernel thuần.
Số c=32 mới không so trực tiếp với c=32/16 slots của các lượt trước.

Chưa đo underrun, tải kéo dài, chất lượng nghe; chưa loại hết nhiễu do clock/thermal
và chưa chứng minh GPU độc quyền chỉ bằng kiểm tra idle. Không quy chênh lệch với
tác giả cho power limit, hoặc chi phí `clone()` cho vài µs, khi chưa profile trực tiếp.

## Lỗi môi trường đã xử lý

Rental Quebec `vk_52920d267668838f` bị loại **trước benchmark**: 120 mẫu trong khoảng
124 giây ghi GPU-Util 50–100%, VRAM 129 MiB khi chưa load model; vài phút sau vẫn 99%.
Đã xuất bằng chứng rồi trả máy, phản hồi VKong **fb-130**. Không dùng số từ rental này.

Trên máy Việt Nam, `attach --setup` báo thiếu `.vkong/init_cmd.sh`; start qua shell
cũng thiếu `.vkong/start.sh`. Dùng bootstrap Python gọi pip rồi start Python trực tiếp
để vượt lỗi, trên cùng rental, phản hồi **fb-131**. Sau setup bỏ bootstrap khỏi source
deploy và không cài lại dependencies giữa các block. Các config stage bỏ `init_cmd`
và giữ chung app name để tái sử dụng máy. Config deploy từng block được lưu cùng log.
Workflow/feedback theo [skill VKong](https://vkong.tli-tech.com/download/skills/vkong-cli).

## Artifacts

- [report.json](report.json): thống kê, method, hash và phần cứng.
- [abba-20261013/](abba-20261013/): từng block có `bench-N.jsonl`, `bench-N.gpu.json`, log server/client, runtime và config deploy.
- [requests-joined.jsonl](abba-20261013/requests-joined.jsonl): từng request đã ghép client/server.
- [source-sha256.json](abba-20261013/source-sha256.json): hash local đối chiếu hash remote trong mỗi `bench-N.gpu.json`.
- Report lượt 3 lần cũ lưu ở `rerun-20261013/`; không dùng làm kết luận hồi quy.
- Đã tải toàn bộ kết quả rồi tắt rental. [Trạng thái sau tắt](abba-20261013/vkong-after-stop.json).
