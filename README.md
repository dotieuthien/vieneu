# VieNeu-TTS inference

[VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) v3 Turbo đọc tiếng Việt theo thời gian thực: gửi một đoạn văn,
khoảng 130 ms sau đã có tiếng, phần còn lại phát tiếp trong lúc model vẫn đang tạo. Model này khá nhỏ. Thứ làm nó
nhanh là cách server chạy model.

Repo này kể lại hai giai đoạn làm việc với server đó. Mọi thứ chạy trên GPU thuê theo giờ qua
[VKong](https://vkong.tli-tech.com), do AI agent điều khiển từ đầu đến cuối: thuê máy, deploy, sửa lỗi, đo, tải kết
quả về, rồi trả máy.

## Câu chuyện

**Giai đoạn 1: hiểu và tạo bản lite (13/10/2026).** Mình dùng agent đọc code của tác giả để hiểu vì sao nó nhanh,
rồi cho agent dựng một bản gọn từ chính code đó, gọi là lite. Đo trên cùng một GPU thì lite chạy ngang SDK gốc, không
nhanh hơn. Tốc độ nằm ở thiết kế của tác giả; bản lite chỉ giúp code dễ đọc hơn. Chi tiết: [PHASE1.md](PHASE1.md).

**Giai đoạn 2: đo rồi mới sửa (14–15/10/2026).** Lần này mình không đoán nữa. Gắn profiler và đo trên GPU thật thì
thấy phần tạo âm thanh đã nhanh, còn chỗ chậm là codec: mỗi lần gọi, CPU mất 182 ms để ra lệnh trong khi GPU chỉ cần
54 ms để tính. Sửa đúng chỗ đó, không đụng vào model, và lite nhanh hơn gấp đôi. Chi tiết:
[OPTIMIZATION.md](OPTIMIZATION.md).

| Giai đoạn 1: VieNeu hoạt động thế nào | Giai đoạn 2: tối ưu codec |
|---|---|
| [![Video giải thích VieNeu](docs/how-it-works.png)](docs/how-it-works.mp4) | [![Video tối ưu codec](docs/optimize.png)](docs/optimize.mp4) |

*Giọng đọc trong cả hai video do chính bản lite tạo ra.*

## Kết quả

Bản lite trước và sau khi tối ưu, đo trên cùng một máy RTX 3060 với 16 ghế (slot):

| | RTF, 1 người | RTF, 16 người | Tiếng đầu, 1 người | Tiếng đầu, 16 người vào cùng lúc |
|---|---:|---:|---:|---:|
| Trước | 0.402 | 0.538 | 124 ms | 294 ms |
| Sau | **0.175** | **0.250** | **76 ms** | 300 ms |

RTF là thời gian xử lý chia cho độ dài âm thanh. Dưới 1 thì người nghe không bị ngắt, càng nhỏ càng nhanh.

- **Nhanh hơn gấp đôi** khi ít người dùng. Tiếng đầu lúc nhiều người vào cùng lúc thì chưa cải thiện.
- **Âm thanh không đổi.** Nghe thử cùng một câu, cùng seed: [trước](docs/samples/original-1.wav) và
  [sau](docs/samples/optimized-1.wav). Đủ 5 câu ở [docs/samples/](docs/samples/).
- **Ổn định khi tải lớn.** Với 32 ghế và 32 người dùng liên tục trong 3 phút: 1824 request, không lỗi nào.

## Chạy thử

Cần CLI của VKong đã đăng nhập. Bước 1 thuê máy, cài thư viện và chạy server:

```bash
cd lite                              # hoặc baseline
vkong validate                       # kiểm tra vkong.yaml
vkong run                            # thuê RTX 3060 12 GB, cài thư viện, chạy serve.py
```

Đợi dòng `Workload running · http://127.0.0.1:PORT`. Đó là địa chỉ của server trên máy bạn. Ở một terminal khác:

```bash
SERVE_URL=http://127.0.0.1:PORT
curl "$SERVE_URL/health"
curl "$SERVE_URL/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"input":"Xin chào.","response_format":"pcm"}' \
  --output speech.pcm                # PCM 16-bit, mono, 48 kHz
```

Sửa code xong thì đẩy lên máy đang thuê, không cần thuê lại. Dùng xong nhớ trả máy, vì `Ctrl+C` chỉ ngắt kết nối và
máy vẫn tính tiền:

```bash
vkong instances                      # lấy <instance-id>, dạng vk_...
vkong attach <instance-id> --update
vkong app stop vieneu-serving-lite   # bản baseline: vieneu-original-baseline
```

`lite/vkong.yaml` đang bật sẵn các cờ dùng để đo (`PROFILE=1 BENCH_MODE=1 CODEC_VERIFY=1`). Khi chạy thật thì bỏ ba cờ
này. Các cờ thường dùng của `serve.py`:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `MAX_STREAMS` | 16 | Số ghế, tức số người được tạo âm thanh cùng lúc. Người thứ 17 phải xếp hàng |
| `OVERLAP_CODEC` | 1 | Codec chạy song song với model. `0` để chạy tuần tự |
| `CODEC_GRAPHS`, `CODEC_RING` | 1, 1 | Hai phần tối ưu của codec. `0` để tắt từng phần |
| `PROFILE` | 0 | Bật profiler: `/profile`, `/profile/trace` (xem OPTIMIZATION.md) |
| `BENCH_MODE` | 0 | Mở `/benchmark` và `/stress` để đo ngay trên máy GPU |
| `WATERMARK` | 1 | Dấu ẩn đánh dấu giọng AI. Các phép đo đều tắt |

Những điều cần biết khi đo trên VKong (tunnel hay đứt, chốt giá trước khi thuê…) nằm ở cuối
[OPTIMIZATION.md](OPTIMIZATION.md#ghi-chú-khi-làm-với-vkong).

## Trong repo

| Đường dẫn | Nội dung |
|---|---|
| [baseline/](baseline/) | SDK gốc của tác giả, bọc thêm HTTP |
| [lite/vieneu_lite/frame.py](lite/vieneu_lite/frame.py) | CUDA graph tạo một mẩu 80 ms cho mọi ghế |
| [lite/vieneu_lite/scheduler.py](lite/vieneu_lite/scheduler.py) | Chia ghế, nạp câu, chạy graph, gọi codec |
| [lite/vieneu_lite/codec.py](lite/vieneu_lite/codec.py) | Codec đã tối ưu (giai đoạn 2) |
| [lite/vieneu_lite/tts.py](lite/vieneu_lite/tts.py) | Cắt câu, phoneme, khoảng nghỉ |
| [lite/vieneu_lite/profiler.py](lite/vieneu_lite/profiler.py) | Profiler, bật bằng `PROFILE=1` |
| [lite/serve.py](lite/serve.py), [lite/benchmark.py](lite/benchmark.py), [lite/stress.py](lite/stress.py) | Server HTTP, client benchmark, bài thử tải và chất lượng |
| [baseline/results/](baseline/results/), [lite/results/](lite/results/) | Số đo gốc của mọi lần chạy; thí nghiệm nào đo cả hai bản thì hai bên cùng cấu trúc |
| [experiments/phase1/](experiments/phase1/) | Tính lại bảng, dựng lại đúng code đã đo và chạy lại giai đoạn 1 (xem [PHASE1.md](PHASE1.md#tự-kiểm-tra-và-chạy-lại)) |
| [docs/](docs/) | Hai video, mẫu nghe, ảnh chụp VKong |
