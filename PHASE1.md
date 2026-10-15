# Giai đoạn 1: hiểu VieNeu và tạo bản lite bằng agent (13/10/2026)

VieNeu-TTS v3 Turbo là một model nhỏ, vậy mà đọc tiếng Việt kịp theo thời gian thực cho nhiều người cùng lúc. Câu hỏi
đầu tiên là: tốc độ đó đến từ đâu? Giai đoạn này đi tìm câu trả lời theo ba bước. Dùng agent đọc code của tác giả.
Cho agent dựng một bản gọn từ chính code đó để hiểu chắc từng phần. Rồi chạy cả hai bản trên GPU thuê để so.

## VieNeu biến chữ thành tiếng như thế nào

[![Video giải thích VieNeu](docs/how-it-works.png)](docs/how-it-works.mp4)

*Video 77 giây. Giọng đọc trong video do chính bản lite tạo ra.*

Đọc code upstream, đường đi của một câu gồm các bước sau:

1. **Chữ thành cách đọc.** Văn bản được đổi sang phoneme trên CPU, rất nhanh.
2. **Tạo âm thanh từng mẩu.** Model tạo âm thanh theo từng mẩu 80 ms. Mỗi mẩu là 16 con số, và mẩu sau được tạo dựa
   trên các mẩu trước, nên phải làm lần lượt (tự hồi quy, AR).
3. **CUDA graph.** Mỗi mẩu cần hàng trăm phép tính nhỏ. Nếu Python ra lệnh từng phép thì GPU chủ yếu ngồi chờ.
   CUDA graph ghi lại cả chuỗi lệnh một lần, sau đó mỗi mẩu chỉ cần một lệnh phát lại.
4. **Nhiều người dùng chung GPU.** Server có 16 ghế (slot). Mỗi lần phát lại tạo một mẩu cho cả 16 ghế. Ai xong thì
   nhường ghế cho người đang chờ.
5. **Codec.** Codec đổi các con số thành sóng âm. Server gom 4 mẩu rồi mới gọi codec một lần. Riêng người mới vào
   được giải mã sớm sau 2 mẩu, để nghe được tiếng đầu nhanh hơn.

Tốc độ nằm ở bước 3, 4 và 5, tức là ở cách server chạy model chứ không ở bản thân model.

## Hai bản để so

- **[baseline](baseline/)** gọi thẳng SDK của tác giả (`Vieneu(...).infer_stream()`), chỉ bọc thêm một lớp HTTP có
  giới hạn số luồng.
- **[lite](lite/)** là bản gọn do agent dựng từ code của tác giả, chỉ giữ phần streaming trên GPU, đủ nhỏ để đọc hết
  trong một buổi. Bản này vẫn dùng SDK để nạp model, codec, phonemizer và các giọng có sẵn.

Ở giai đoạn này, lite gồm ba file chính:

| File | Việc |
|---|---|
| [frame.py](lite/vieneu_lite/frame.py) | Một CUDA graph cho mỗi mẩu 80 ms: chọn 16 con số, chống lặp, sampling riêng từng request |
| [scheduler.py](lite/vieneu_lite/scheduler.py) | 16 ghế dùng chung GPU: nhận request, nạp câu, chạy graph, gom 4 mẩu rồi gọi codec |
| [tts.py](lite/vieneu_lite/tts.py) | Cắt văn bản thành câu, đổi sang phoneme, chèn khoảng nghỉ giữa các câu |

Lite bỏ những đường không dùng cho GPU streaming (CPU/ONNX, GGUF, Gradio) và thêm vài chốt an toàn. Tham số sampling
sai bị từ chối trước khi vào GPU. Lỗi của một request không làm chết server. Còn `/health` trả 503 khi worker đã chết.

## Chạy trên VKong cùng agent

[VKong](https://vkong.tli-tech.com) cho thuê GPU theo giờ và điều khiển hoàn toàn bằng CLI. Vì vậy AI agent tự làm được
trọn vòng: viết code, thuê máy bằng `vkong run`, gọi API qua tunnel về máy mình, chạy benchmark ngay trên máy GPU, tải
log về, rồi trả máy. Người chỉ cần đọc kết quả và quyết định bước tiếp theo. Một RTX 3060 có giá khoảng
$0.06–0.10/giờ. Các lệnh cụ thể nằm ở phần [Chạy thử](README.md#chạy-thử) của README.

Ảnh chụp dashboard VKong trong lần chạy lite để thu giọng cho video:

![Tiến trình chạy app vieneu-serving-lite trên VKong](docs/vkong_app.png)

*Thuê máy 29 giây, đồng bộ code 5 giây, cài thư viện 1 phút 20 giây, rồi chạy server.*

![Log server khi đọc lời cho video](docs/logs.png)

*Mỗi câu lời thoại là một request, gửi lần lượt. Tiếng đầu ra sau 181–205 ms (đo phía server), RTF 0.66–0.71.*

![Biểu đồ GPU, CPU và RAM của máy](docs/metrics.png)

*GPU, CPU và RAM của máy trong lần chạy đó.*

## Kết quả: lite chạy ngang SDK gốc

### Phương pháp đo

**Nơi chạy client đo.** Client đo chạy trên chính máy GPU, không chạy trên máy điều khiển. Máy điều khiển chỉ gửi lệnh
`POST /benchmark` qua kết nối của VKong. Khi nhận lệnh, server khởi chạy `benchmark.py` thành một tiến trình con trên
cùng máy. Tiến trình này gửi request tới `http://127.0.0.1:8000` và ghi kết quả ra file trên máy GPU. Máy điều khiển
tải file đó về sau khi đo xong. Như vậy, độ trễ mạng giữa máy điều khiển và máy GPU không nằm trong số đo nào. Dòng đầu
của mỗi `bench-<c>.log` ghi lại nơi chạy client, ví dụ
`benchmark_host=00e188f076e2 server_pid=647 base=http://127.0.0.1:8000`.

**Máy đo.** Hai bản chạy trên cùng một máy thuê, cùng container `00e188f076e2` và cùng GPU (UUID
`GPU-6f4f1fdb-b8d6-e268-3730-8d9b7edd0919`):

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, driver 595.104.02, giới hạn công suất 120 W |
| Máy thuê | VKong `vk_7fba8887e3fa2c37`, $0.078/giờ; cấu hình thuê 4 nhân CPU, 16 GB RAM |
| Phần mềm | Image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`; torch 2.8.0+cu128, vieneu 3.8.3, transformers 4.57.6 |
| Xung nhịp GPU | Không khóa được, vì máy thuê từ chối `nvidia-smi -lgc`. Trong lúc đo: 1777–1935 MHz, 53–71 °C |

**Cấu hình.** Hai bản dùng cùng cấu hình: 16 ghế (riêng mức 32 người dùng 32 ghế), gọi codec mỗi 4 mẩu, giải mã sớm
sau 2 mẩu, tắt watermark, `OMP_NUM_THREADS=4`. Bản lite được đo là bản đầu tiên, khi codec còn chạy lần
lượt ngay trong vòng lặp của model.

**Tải.** Mọi request đọc cùng một câu, dài khoảng 4 giây âm thanh, bằng giọng mặc định Hải Đăng:

> Xin chào, đây là giọng đọc VieNeu phát trực tiếp qua API chuẩn OpenAI.

Mỗi lượt gồm `c` request, và `c` client bắt đầu cùng lúc. Trước mỗi mức tải, client đợi GPU rảnh (mức sử dụng ≤ 5%
trong 5 giây liên tiếp), rồi chạy 2 lượt khởi động không tính vào kết quả. Trước mỗi lượt, seed được đặt lại thành
42 + số thứ tự lượt. Chỉ một request lỗi là cả lần đo dừng. Lần này 1260/1260 request đều thành công.

**Thứ tự A–B–B–A.** Có 4 block theo thứ tự SDK, lite, lite, SDK, cùng trên một máy. Mỗi block deploy lại code bằng
`vkong attach --update`, tức là khởi động lại server, rồi đo:

- 16 ghế: 1 người dùng × 15 lượt; 4, 8 và 16 người dùng × 5 lượt mỗi mức.
- Khởi động lại với 32 ghế: 32 người dùng × 5 lượt.

Mỗi bản có 630 request đo: 30, 40, 80, 160 và 320 request ở từng mức.

**Thước đo.** Cả hai càng thấp càng tốt.

- **TTFA**: đo ở client, từ lúc gửi request đến khi nhận byte âm thanh đầu tiên.
- **RTF**: thời gian từ lúc gửi request đến khi nhận hết âm thanh, chia cho độ dài âm thanh. Dưới 1 thì người nghe
  không bị ngắt.

Server cũng ghi TTFA phía server cho cùng các request đó, tính từ lúc handler nhận request đến chunk âm thanh đầu tiên.
Số này nằm trong `server-requests.jsonl`.

**Thống kê.** Mỗi request là một mẫu. Mọi block của cùng một bản được gộp lại. Số trong bảng là trung vị, nội suy
tuyến tính, không lấy trung vị của các trung vị. Mỗi bản chỉ có hai block, nên đây là so sánh mô tả, không phải kiểm
định thống kê.

### Lite so với SDK gốc

| Người dùng cùng lúc | TTFA SDK gốc | TTFA lite | RTF SDK gốc | RTF lite |
|---:|---:|---:|---:|---:|
| 1 | 127.0 ms | 129.1 ms | 0.529 | 0.548 |
| 4 | 174.4 ms | 173.6 ms | 0.568 | 0.568 |
| 8 | 202.6 ms | 202.5 ms | 0.591 | 0.590 |
| 16 | 243.6 ms | 273.8 ms | 0.674 | 0.659 |
| 32 (32 ghế) | 588.4 ms | 591.5 ms | 1.040 | 1.039 |

Số liệu: [baseline/results/abba-20261013/](baseline/results/abba-20261013/) (block 1 và 4) và
[lite/results/abba-20261013/](lite/results/abba-20261013/) (block 2 và 3).

### Giới hạn

- Chỉ đo một câu ngắn, gửi dồn cùng lúc. Văn bản dài nhiều câu, hay request đến rải rác, chưa được đo.
- Xung nhịp GPU không khóa được, và mỗi máy thuê một khác. Số tuyệt đối chỉ đúng cho máy đã đo; phép so sánh chỉ có ý
  nghĩa giữa các bản đo trong cùng một lần chạy.

## Tự kiểm tra và chạy lại

### Kiểm tra số trong bảng (không cần GPU)

```bash
python3 experiments/phase1/analyze.py
```

Lệnh này in lại bảng ở trên, tính từ file đo thô. Mỗi request đã đo là một dòng trong file `bench-<c>.jsonl`,
với `c` là số người dùng cùng lúc. Các file này nằm trong `baseline/results/abba-20261013/` (SDK) và
`lite/results/abba-20261013/` (lite), mỗi thư mục `block-*` là một lần bật server. Số trong bảng là trung vị của mọi
dòng.

### Chạy lại trên VKong

Cách làm: thuê **một** máy, chạy SDK và đo. Sau đó đổi code sang lite ngay trên máy đó và đo tiếp, theo thứ tự SDK →
lite → lite → SDK. Đo trên cùng một máy thì hai bản mới so được với nhau. Mất khoảng 30 phút máy, dưới $0.05.

**1. Lấy đúng code lúc đo**

```bash
python3 experiments/phase1/stage.py abba /tmp/p1
```

Lệnh này tạo hai thư mục `/tmp/p1/abba/sdk` và `/tmp/p1/abba/lite`, mỗi thư mục có sẵn `vkong.yaml`. Bản lite ở đây cũ
hơn thư mục `lite/` hiện tại, vì chưa có phần tối ưu của giai đoạn 2. Nếu code dựng ra giống hệt code đã đo, lệnh in
`all sources match the original run`.

**2. Thuê máy và chạy SDK**

```bash
vkong run -C /tmp/p1/abba/sdk --detach    # in giá, hỏi xác nhận, thuê máy, cài thư viện, bật server
vkong instances                           # lấy id của máy, dạng vk_xxxx
vkong attach vk_xxxx -C /tmp/p1/abba/sdk  # mở kết nối, in ra http://127.0.0.1:PORT
```

Để cửa sổ `attach` mở. Các lệnh sau gõ ở một cửa sổ khác.

**3. Đo, ngay trên máy GPU**

Lệnh `POST /benchmark` không tự đo. Nó bảo server chạy client đo ngay trên máy GPU, như mô tả ở
[Phương pháp đo](#phương-pháp-đo). Máy bạn chỉ chờ xong rồi tải file kết quả về.

```bash
URL=http://127.0.0.1:PORT
OUT=baseline/results/abba-lan2/block-1-slots-16
mkdir -p $OUT

curl -X POST "$URL/benchmark?concurrency=1&repeats=15"   # bảo máy GPU bắt đầu đo 1 người dùng, 15 lượt
curl "$URL/benchmark"                                     # hỏi lại tới khi thấy "running": false
curl -o $OUT/bench-1.jsonl "$URL/results/bench-1.jsonl"   # tải kết quả về
```

Làm lại ba lệnh `curl` với `concurrency=4`, `8`, `16` và `repeats=5`.

**4. Đổi sang lite trên cùng máy**

Nhấn Ctrl+C ở cửa sổ `attach`. Lệnh này chỉ đóng kết nối, máy vẫn chạy. Sau đó:

```bash
vkong attach vk_xxxx --update -C /tmp/p1/abba/lite
```

`--update` gửi thư mục `lite` lên đúng máy đó và bật lại server bằng code mới. Thư viện đã cài ở bước 2 nên không cài
lại. Đo như bước 3, nhưng lưu vào `lite/results/abba-lan2/block-2-slots-16`.

**5. Đủ thứ tự SDK → lite → lite → SDK**

Làm lại bước 4 thêm hai lần: với `lite` (lưu vào `block-3-slots-16`), rồi với `sdk` (lưu vào
`baseline/results/abba-lan2/block-4-slots-16`).

Bảng có thêm một dòng 32 người dùng với 32 ghế. Để đo dòng này, sửa `MAX_STREAMS=16` thành `MAX_STREAMS=32` trong dòng
`start` của `vkong.yaml`, chạy lại `vkong attach ... --update`, rồi đo `concurrency=32&repeats=5` và lưu vào
`block-<n>-slots-32`.

**6. Trả máy và tính bảng**

```bash
vkong app stop vieneu-phase1 --yes        # Ctrl+C không trả máy; máy tính tiền tới khi stop
python3 experiments/phase1/analyze.py --abba abba-lan2
```

**Lưu ý**

- Mỗi máy thuê một khác, nên số của bạn sẽ không giống hệt bảng. Thứ cần so là chênh lệch giữa SDK và lite trong
  cùng một lần chạy.
- Kết nối hay đứt (`Connection to the machine was lost`) trong khi máy vẫn chạy. Gõ `vkong instances` để chắc máy còn,
  rồi `vkong attach vk_xxxx -C <thư mục>` để nối lại. Không có `--update` thì code và server giữ nguyên.
