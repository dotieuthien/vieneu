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

Hai thước đo, cả hai càng thấp càng tốt:

- **TTFA**: thời gian từ lúc gửi yêu cầu đến khi nhận được tiếng đầu tiên.
- **RTF**: tổng thời gian xử lý chia cho độ dài âm thanh. Dưới 1 thì người nghe không bị ngắt.

Mọi số trong bảng là trung vị trên từng request, đo trên RTX 3060. Client chạy ngay trên máy GPU, nên không tính
thời gian qua tunnel. Cả hai bản dùng 16 ghế, gọi codec mỗi 4 mẩu, giải mã sớm sau 2 mẩu, và tắt watermark.

**Lite so với SDK gốc.** Đo xen kẽ theo thứ tự A–B–B–A, mỗi bản 630 request:

| Người dùng cùng lúc | TTFA SDK gốc | TTFA lite | RTF SDK gốc | RTF lite |
|---:|---:|---:|---:|---:|
| 1 | 127.0 ms | 129.1 ms | 0.529 | 0.548 |
| 4 | 174.4 ms | 173.6 ms | 0.568 | 0.568 |
| 8 | 202.6 ms | 202.5 ms | 0.591 | 0.590 |
| 16 | 243.6 ms | 273.8 ms | 0.674 | 0.659 |
| 32 (32 ghế) | 588.4 ms | 591.5 ms | 1.040 | 1.039 |

**Thử cho codec chạy song song với model** (`OVERLAP_CODEC=1`). Đây là một phép thử nhỏ, mỗi cấu hình 6 request với 1
người và 32 request với 16 người:

| Người dùng cùng lúc | Cấu hình | TTFA | RTF |
|---:|---|---:|---:|
| 1 | SDK gốc | 131.1 ms | 0.535 |
| 1 | Lite, chạy tuần tự | 131.7 ms | 0.552 |
| 1 | Lite, codec chạy song song | 151.0 ms | 0.497 |
| 16 | SDK gốc | 272.9 ms | 0.730 |
| 16 | Lite, chạy tuần tự | 233.9 ms | 0.680 |
| 16 | Lite, codec chạy song song | 319.4 ms | 0.652 |

Mọi request đọc cùng một câu, dài khoảng 4 giây âm thanh, bằng giọng mặc định Hải Đăng:

> Xin chào, đây là giọng đọc VieNeu phát trực tiếp qua API chuẩn OpenAI.

Vì vậy các số trên chỉ nói về câu ngắn gửi dồn cùng lúc. Văn bản dài nhiều câu, hay request đến rải rác, chưa được đo.

## Rút ra, và vì sao có giai đoạn 2

- **Bản gọn hơn không chạy nhanh hơn.** Lite chạy xấp xỉ SDK gốc ở mọi mức tải. Những thứ làm nên tốc độ (CUDA
  graph, nhiều người chung một lượt chạy, codec theo từng đoạn) đều là thiết kế của tác giả, và lite giữ nguyên chúng.
- **Đoán thì khó trúng.** Cho codec chạy song song là một phép thử theo cảm giác. Cả câu xong sớm hơn một chút, nhưng
  tiếng đầu lại đến chậm hơn.

Muốn nhanh hơn thật thì trước hết phải biết thời gian đi đâu. Giai đoạn 2 bắt đầu bằng việc đo:
[OPTIMIZATION.md](OPTIMIZATION.md).
