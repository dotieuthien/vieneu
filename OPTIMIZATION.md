# Giai đoạn 2: tìm chỗ chậm và tối ưu (14–16/10/2026)

Hết [giai đoạn 1](PHASE1.md), bản lite chạy ngang SDK gốc. Giai đoạn này làm theo thứ tự: bật profiler, đo, tìm chỗ
chậm, đọc code để hiểu vì sao, sửa, kiểm tra âm thanh không đổi, rồi đo lại so với SDK. Model giữ nguyên: trọng số,
kiến trúc, kiểu số và cách sampling đều không đổi.

Video tóm tắt: [docs/optimize.mp4](docs/optimize.mp4).

## Kết quả

Bản lite đã tối ưu so với SDK gốc, cùng một máy RTX 3060 12 GB:

| Người dùng cùng lúc | TTFA P50 / P90, SDK gốc | TTFA P50 / P90, bản tối ưu | RTF P50 / P90, SDK gốc | RTF P50 / P90, bản tối ưu |
|---:|---:|---:|---:|---:|
| 1 | 110.1 / 110.8 ms | **62.7 / 63.4 ms** | 0.476 / 0.480 | **0.162 / 0.167** |
| 4 | 151.2 / 152.5 ms | **116.0 / 120.9 ms** | 0.510 / 0.525 | **0.184 / 0.191** |
| 8 | 177.0 / 186.7 ms | **143.1 / 154.5 ms** | 0.533 / 0.548 | **0.196 / 0.202** |
| 16 | 205.3 / 343.8 ms | 210.9 / **226.2 ms** | 0.591 / 0.607 | **0.217 / 0.224** |
| 32 (32 ghế) | 494.9 / 672.5 ms | **411.6 / 458.5 ms** | 0.918 / 0.955 | **0.347 / 0.364** |

- RTF giảm 2.6–2.9 lần ở mọi mức tải.
- Tiếng đầu với 1 người dùng: 110 → 63 ms.
- 16 người vào cùng lúc: TTFA P50 ngang nhau, P90 giảm từ 344 xuống 226 ms.
- Âm thanh không đổi (xem bước 5).

Cách đo và máy đo: [bước 8](#bước-8-bench-so-với-sdk-gốc).

## Các cờ dùng khi đo

Các cờ của `lite/serve.py`, ghi vào dòng `start` trong `vkong.yaml`:

| Cờ | Dùng để |
|---|---|
| `PROFILE=1` | Bật profiler: `/profile`, `/profile/trace`, frame anatomy trong log, watchdog |
| `BENCH_MODE=1` | Mở `/benchmark` và `/stress` để tạo tải ngay trên máy GPU |
| `CODEC_VERIFY=1` | Lúc khởi động, so codec mới với codec gốc của SDK |
| `CODEC_GRAPHS=0`, `CODEC_RING=0` | Tắt từng thay đổi của codec để so |
| `MAX_STREAMS` | Số ghế |
| `WATERMARK=0` | Tắt watermark, như mọi lần đo |

## Bước 1. Viết profiler

Mục tiêu: trả lời "thời gian đi đâu" trước khi sửa bất cứ thứ gì.

[profiler.py](lite/vieneu_lite/profiler.py) gắn vào scheduler, codec và các thread nhận request. Bật bằng `PROFILE=1`,
tắt thì không tốn gì. Nó cho biết:

| Chỉ số | Ý nghĩa |
|---|---|
| `tick.period` | Thời gian model tạo một mẩu cho mọi ghế. Phải dưới 80 ms mới kịp thời gian thực |
| `codec.decode` | Thời gian CPU và GPU của mỗi lần gọi codec |
| `decode.backpressure` | Thời gian model phải dừng vì codec chưa xong |
| `req.first_audio` | Thời gian tới tiếng đầu, tách theo từng chặng: chờ, nạp câu, 2 mẩu đầu, codec |
| Frame anatomy | In ra log lúc khởi động: một mẩu chạy thường cần bao nhiêu lệnh, chạy bằng CUDA graph cần bao nhiêu |
| Trace | `torch.profiler` trên mọi thread, lưu file mở bằng [Perfetto](https://ui.perfetto.dev) |
| Watchdog | Có người dùng mà 30 giây không có mẩu nào được tạo thì in stack của mọi thread |

Hai lỗi đã gặp khi viết, sửa sẵn trong code:

- `torch.profiler` mặc định chỉ ghi thread gọi nó, nên trace của worker và codec bị trống. Phải bật
  `profile_all_threads=True`.
- Lite tự capture CUDA graph, nên lần trace thứ hai có thể crash. `PROFILE=1` tự đặt `TEARDOWN_CUPTI=0`.

**Cách dùng.** Deploy bản lite cần đo với `PROFILE=1 BENCH_MODE=1` trong dòng `start` của `vkong.yaml`, rồi mở kết nối:

```bash
vkong run -C <thư mục lite> --detach
vkong instances
vkong attach vk_xxxx -C <thư mục lite>
```

Ở cửa sổ khác, tạo tải 1 người dùng ngay trên máy GPU rồi đọc profiler:

```bash
URL=http://127.0.0.1:PORT
curl -X POST "$URL/benchmark?concurrency=1&repeats=15"
curl -X POST "$URL/profile/reset"
curl "$URL/benchmark"
curl "$URL/profile" > profile-c1-notrace.txt
```

- Gọi `profile/reset` sau khi `curl "$URL/results/bench-1.log"` đã có dòng `measured`, để bỏ qua lúc khởi động.
- Gọi lại `curl "$URL/benchmark"` tới khi thấy `"running": false`, rồi mới lấy `/profile`.

Lấy trace 2 giây trong lúc có tải:

```bash
curl -X POST "$URL/benchmark?concurrency=1&repeats=15"
curl -X POST "$URL/profile/trace?seconds=2"
curl -O "$URL/results/profile/trace-<thời gian>.json.gz"
```

Lệnh trace in ra tên file. Mở file bằng [Perfetto](https://ui.perfetto.dev): bấm Open trace file, chọn thẳng file
`.json.gz`. Đổi `concurrency=1&repeats=15` thành `concurrency=16&repeats=8` để đo với 16 người dùng.

## Bước 2. Đo bản cũ, tìm chỗ chậm

Mục tiêu: đo bản lite cũ với 1 người dùng, xem chỗ nào tốn thời gian.

**Kết quả** ([lite/results/profile-20261014/](lite/results/profile-20261014/)):

| Chỉ số | Giá trị | Đọc ra |
|---|---:|---|
| `tick.period` | 11.8 ms | Model tạo một mẩu nhanh gấp ~7 lần thời gian thực. Model không chậm |
| `codec.decode`, CPU | 182 ms | Mỗi lần gọi codec (4 mẩu, 320 ms âm thanh) tốn 182 ms CPU |
| `codec.decode`, GPU | 183 ms | Tính từ lúc bắt đầu tới lúc xong trên GPU, gồm cả lúc GPU chờ lệnh |
| `decode.backpressure` | 129 ms | Model phải dừng chờ codec |
| `req.first_audio` | 167 ms | Tiếng đầu, đo phía server |
| GPU bận (trace) | 20% | GPU phần lớn thời gian ngồi chờ |

Trong trace, lần gọi 4 mẩu có khoảng 3500 lệnh gửi xuống GPU. Trung bình mỗi lần gọi, GPU chỉ chạy kernel khoảng
54 ms.

**Kết luận:** chỗ chậm là codec, và nó chậm vì CPU gửi lệnh không kịp, không phải vì GPU tính lâu.

Bản cũ lúc đó là lite ở commit `e4ef962` gắn thêm profiler. Phần gắn thêm này không được commit riêng, nên không
deploy lại đúng bản đó được; kết quả của nó nằm trong thư mục ở trên. Lần đo này, tải được gửi từ máy điều khiển qua kết
nối VKong; các số ở bảng đều đo phía server nên không bị ảnh hưởng.

## Bước 3. Đọc code codec, tìm nguyên nhân

Mục tiêu: hiểu vì sao codec gửi nhiều lệnh như vậy.

Đọc source `vieneu==3.8.3` và code của codec
([MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)). Codec nhận 16 số mỗi
mẩu và chạy qua 12 lớp transformer để ra 80 ms sóng âm. Mỗi lớp nhìn lại 4–10 giây trước đó, giữ trong KV cache.

Ba nguyên nhân, mỗi cái kiểm tra lại bằng trace:

| Nguyên nhân | Trong code | Bằng chứng trong trace |
|---|---|---|
| Giải mã từng mẩu một | `DecodeSession.step` chạy hết 12 lớp cho mẩu 1, rồi mới tới mẩu 2 | Lần gọi 4 mẩu tốn gấp đôi lần gọi 2 mẩu: 68 so với 34 phép convolution, 155 so với 91 ms CPU |
| CPU đợi GPU sau mỗi mẩu | `.item()` đọc độ dài âm thanh về CPU sau mỗi mẩu | Mỗi mẩu thêm một lần `aten::item` |
| KV cache bị chép lại | Ở mỗi lớp, mỗi mẩu: `cat` → cắt → `where` → `copy_`, chép cả cửa sổ 4 lần để thêm 1 bước | Lần gọi 4 mẩu có 48 lần attention, đi kèm 241 `cat`, 304 `where`, 604 `copy_` |

Số trong cột cuối đếm từ [trace-20261014-104645.json.gz](lite/results/profile-20261014/trace-20261014-104645.json.gz).

Code của tác giả có chế độ `use_cuda_graph` cho codec, kèm ghi chú "rẻ hơn 3 lần nhưng ra âm thanh sai". Lý do: `step`
giữ con trỏ tới vùng nhớ output của graph, và lần phát sau ghi đè lên lần trước.

## Bước 4. Sửa codec

Mục tiêu: bỏ phần CPU gửi lệnh, không đổi kết quả. Code mới: [codec.py](lite/vieneu_lite/codec.py).

| Thay đổi | Trước | Sau | Tác dụng |
|---|---|---|---|
| 1. Số lượt tính | 4 mẩu tính 4 lượt nối tiếp | 4 mẩu tính chung 1 lượt | CPU mỗi lần gọi còn ~40 ms |
| 2. Cách gửi lệnh | ~3500 lệnh nhỏ mỗi lần gọi | CUDA graph: 1 lệnh | CPU còn ~1.3 ms |
| 3. KV cache | Chép lại cả cửa sổ 4 lần mỗi bước | Mảng vòng: ghi đè 1 ô | GPU còn ~22 ms |

- **Tính chung 1 lượt** được vì các con số của cả 4 mẩu đã có sẵn từ model. Chỉ cần che để mỗi mẩu không nhìn thấy
  mẩu sau nó. Độ dài âm thanh cũng biết trước, nên bỏ được `.item()`.
- **CUDA graph:** mỗi độ dài 1–4 mẩu có một graph. Output được chép ra trước lần phát sau, tránh lỗi của chế độ graph
  gốc.
- **Mảng vòng:** xem ví dụ ngay dưới.

**KV cache trước và sau, ví dụ thu nhỏ.** Cửa sổ 6 bước, mỗi lần gọi thêm 2 bước. Ví dụ là lần gọi thêm bước 10 và 11.

*Trước:* cache dài đúng 6 ô, luôn xếp từ cũ tới mới. Để thêm 2 bước, code gốc làm 4 thao tác, thao tác nào cũng chép
cả mảng:

| Thao tác | Mảng sau thao tác | Số ô phải chép |
|---|---|---:|
| Cache trước lần gọi | 4 5 6 7 8 9 | — |
| 1. `cat`: nối bước mới vào cuối | 4 5 6 7 8 9 **10 11** | 8 |
| 2. Cắt, giữ 6 bước cuối | 6 7 8 9 10 11 | 6 |
| 3. `where`: giữ cache cũ cho người không chạy lượt này | 6 7 8 9 10 11 | 6 |
| 4. `copy_`: ghi vào cache | 6 7 8 9 10 11 | 6 |
| **Tổng** | | **26 ô để thêm 2 bước** |

Việc này lặp lại cho key, value và vị trí, ở mỗi lớp, mỗi mẩu. Cửa sổ thật lớn hơn ví dụ rất nhiều:

| Tầng | Số lớp | Bước mỗi mẩu 80 ms | Cửa sổ | Mảng vòng (cửa sổ + 4 mẩu) |
|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 10 giây = 500 bước | 516 ô |
| 2 | 2 | 8 | 8 giây = 800 bước | 832 ô |
| 3 | 2 | 16 | 6 giây = 1200 bước | 1264 ô |
| 4 | 4 | 32 | 4 giây = 1600 bước | 1728 ô |

Ở tầng 4, mỗi mẩu thêm 32 bước nhưng code gốc phải chép khoảng 6400 ô (1632 + 3 × 1600). Mảng vòng chỉ ghi 32 ô. Số
bước và cửa sổ tính từ `config.json` của codec.

*Sau:* mảng vòng có 6 + 2 = 8 ô. Bước số `p` luôn nằm ở ô `p mod 8`. Chữ đậm là bước vừa ghi:

| Ô | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Trước lần gọi | 8 | 9 | 2 | 3 | 4 | 5 | 6 | 7 |
| Ghi bước 10, 11 | 8 | 9 | **10** | **11** | 4 | 5 | 6 | 7 |
| Lần gọi sau: ghi bước 12, 13 | 8 | 9 | 10 | 11 | **12** | **13** | 6 | 7 |

- Mỗi lần gọi chỉ ghi đúng 2 ô, không chép lại ô nào.
- Bước 10, 11 ghi đè lên bước 2, 3. Các bước này đã ra khỏi cửa sổ: bước 10 chỉ cần nhìn lại tới bước 5.

*Lúc tính attention:* cache cũ đã xếp sẵn theo thứ tự, nên đọc thẳng. Mảng vòng thì không sắp lại; mỗi ô được tính ra
nó đang chứa bước số mấy, rồi mask chỉ cho mỗi bước nhìn 6 bước gần nhất, tính tới chính nó. Ví dụ với bước 11:

| Ô | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Đang chứa bước | 8 | 9 | 10 | 11 | 4 | 5 | 6 | 7 |
| Bước 11 nhìn thấy | ✓ | ✓ | ✓ | ✓ | · | · | ✓ | ✓ |

Bước 11 nhìn bước 6 tới 11, giống hệt cache cũ, dù các bước này nằm rải ở ô 6, 7, 0, 1, 2, 3.

**Tiết kiệm được bao nhiêu với cửa sổ thật.** Dữ liệu KV phải ghi cho mỗi lần gọi codec (4 mẩu), cộng qua cả 12 lớp.
Mỗi ô là key + value của một bước, 2 KB. Cả codec gốc lẫn codec mới đều tính đủ mọi ghế dành sẵn (`MAX_STREAMS`), kể cả
ghế trống, nên với 16 ghế thì con số là cột cuối, dù chỉ có 1 người dùng:

| Cách làm | Ô phải ghi, mỗi ghế | Dữ liệu, mỗi ghế | Dữ liệu, 16 ghế |
|---|---:|---:|---:|
| Codec gốc, giải mã từng mẩu | ~199 nghìn | ~408 MB | ~6.5 GB |
| Gom 4 mẩu, KV vẫn kiểu cũ | ~50 nghìn | ~103 MB | ~1.65 GB |
| Gom 4 mẩu + mảng vòng | 768 | ~1.6 MB | ~25 MB |

Đây là lượng dữ liệu phải ghi, không phải bộ nhớ chiếm. KV của codec chỉ chiếm khoảng 25 MB mỗi ghế (khoảng 406 MB cho
16 ghế); code gốc ghi đi ghi lại cùng vùng nhớ đó nhiều lần trong mỗi lần gọi.

Đo thực tế (16 ghế, cùng một máy, bảng ở bước 7), so "gom + CUDA graph" với "gom + CUDA graph + mảng vòng":

| Số người dùng | GPU mỗi lần gọi codec | RTF P50 |
|---|---:|---:|
| 1 người dùng | 32.1 → 22.3 ms | 0.209 → 0.175 |
| 16 người dùng | 33.1 → 22.1 ms | 0.307 → 0.250 |

- Vì codec luôn tính đủ 16 ghế, mức tiết kiệm gần như không đổi giữa 1 và 16 người dùng.
- Bớt 1.65 GB ghi, cộng phần đọc tương đương, với băng thông bộ nhớ 360 GB/s của RTX 3060, là khoảng 9–10 ms. Con số
  này khớp với khoảng 10 ms đo được.
- Mảng vòng chỉ bỏ phần chép. Attention vẫn đọc cả cửa sổ ở mỗi lần gọi, ở cả hai cách.

Scheduler cũng sửa hai chỗ sau khi đo lại:

- Khi có người mới vào giữa chừng, chỉ giải mã sớm cho người mới, không giải mã luôn người cũ.
- Mỗi tick nhận tối đa 16 người thay vì 8, để 16 người vào cùng lúc không phải nạp câu hai đợt.

## Bước 5. Kiểm tra âm thanh không đổi

Mục tiêu: chứng minh codec mới cho cùng kết quả trước khi đo tốc độ.

- **So codec mới với codec gốc.** Hàm `verify()` cho cùng một chuỗi mã đi qua codec gốc của SDK và codec mới: hai người
  dùng, dài khoảng 13 giây để mảng vòng quay hết, có người vào giữa chừng, có người rời đi. Chạy bằng `CODEC_VERIFY=1`
  lúc server khởi động, kết quả nằm trong `curl "$URL/health"`, mục `codec_check`. Lần đo ngày 16/10: tương quan
  0.9999992, sai lệch lớn nhất 0.0105 trên thang ±1
  ([runtime.json](lite/results/profile-20261016/runtime.json)).
- **Mẫu nghe.** Khởi động lại từng bản, gọi `curl -X POST "$URL/benchmark/seed?seed=42"` trước mỗi câu, đọc 5 câu.
  Model sinh cùng một chuỗi mã, nên khác biệt chỉ nằm ở codec: tương quan 0.9999998, SNR khoảng 64 dB. Nghe ở
  [docs/samples/](docs/samples/), ví dụ [original-1.wav](docs/samples/original-1.wav) và
  [optimized-1.wav](docs/samples/optimized-1.wav).

Không so được dạng sóng của hai lần chạy model khác nhau: model chạy bf16, nên mã sinh ra rẽ nhánh ngay từ vài mẩu
đầu. Muốn so chất lượng lúc tải nặng thì dùng nhận dạng giọng nói (bước 9).

## Bước 6. Debug khi GPU treo

Lần đầu deploy codec mới, server không lên: không lỗi, không log.

1. `serve.py` bật `faulthandler`: khởi động quá 60 giây thì in stack của mọi thread vào log.
2. Stack cho thấy cả hai thread đều đang chờ GPU: thread codec ở `audio.cpu()`, thread model ở
   `torch.cuda.Event.record`. Tức là GPU kẹt thật.
3. Nguyên nhân (suy ra từ việc bản sửa có tác dụng): graph của model và graph của codec được capture trên cùng một
   stream, nên dùng chung vùng nhớ nháp của cuBLAS. Khi hai graph chạy song song thì kẹt.
4. Sửa: capture graph của codec trên stream riêng. Hết treo.

Sau đó thêm watchdog (`PROFILE=1`): có người dùng mà 30 giây không có mẩu nào được tạo thì in stack mọi thread.

Một lần đi sai: kết nối VKong đứt, `/health` không trả lời, và mình kết luận GPU treo. Nhưng máy vẫn `running`, chỉ
kết nối bị đứt. Khi nghi GPU treo: chạy `vkong instances` để xem máy còn chạy không, nối lại bằng `vkong attach`, rồi
tìm dòng `no frame for` của watchdog trong log.

## Bước 7. Profile lại sau khi sửa

Deploy bản tối ưu (commit `60d316a`) với cấu hình đã lưu ở
[profile-20261016/vkong-deployed.yaml](lite/results/profile-20261016/vkong-deployed.yaml) (có `PROFILE=1 BENCH_MODE=1
CODEC_VERIFY=1`), rồi chạy các lệnh ở bước 1 với 1 và 16 người dùng.

| Chỉ số, 1 người dùng | Trước | Sau |
|---|---:|---:|
| Mỗi lần gọi codec: CPU | 182 ms | 1.2 ms |
| Mỗi lần gọi codec: GPU | 183 ms | 22 ms |
| Model phải chờ codec | 129 ms | 0 ms |
| Tạo một mẩu cho mọi ghế | 11.8 ms | 9.1 ms |
| GPU bận (trace) | 20% | 70% |
| Tiếng đầu, đo phía server | 167 ms | 65 ms |

"Trước" đo ngày 14/10, "sau" đo ngày 16/10 trên hai máy khác nhau. Số phía CPU phụ thuộc máy, nên chỉ nên so độ lớn.

Trace để mở bằng Perfetto:

- Trước: [profile-20261014/run2/trace-20261014-110031.json.gz](lite/results/profile-20261014/run2/trace-20261014-110031.json.gz)
- Sau, 1 người dùng: [profile-20261016/trace-c1.json.gz](lite/results/profile-20261016/trace-c1.json.gz)
- Sau, 16 người dùng: [profile-20261016/trace-c16.json.gz](lite/results/profile-20261016/trace-c16.json.gz)

Trong trace "sau", mỗi lần gọi codec chỉ còn 11 lệnh gửi xuống GPU. Bảng profiler đầy đủ nằm trong
`profile-c1-notrace.txt` và `profile-c16-notrace.txt` của thư mục [profile-20261016/](lite/results/profile-20261016/).

**Mỗi thay đổi giúp bao nhiêu.** Đo trên một máy khác ngày 14/10, 16 ghế, bật tắt từng thay đổi bằng
`CODEC_GRAPHS` và `CODEC_RING`:

| Cấu hình | RTF P50, 1 người | RTF P50, 16 người | TTFA P50, 1 người | Codec mỗi lần gọi: CPU / GPU |
|---|---:|---:|---:|---:|
| Bản cũ | 0.402 | 0.538 | 123.8 ms | — |
| 1. Tính chung 1 lượt | 0.208 | 0.323 | 111.4 ms | 40 ms / — |
| 1 + 3. Thêm mảng vòng | 0.181 | 0.289 | 105.1 ms | 34 ms / — |
| 1 + 2. Thêm CUDA graph | 0.209 | 0.307 | 88.4 ms | 1.3 ms / 32 ms |
| 1 + 2 + 3. Đủ cả ba | **0.175** | **0.250** | **76.0 ms** | 1.2 ms / 22 ms |

- Tính chung 1 lượt cho phần lớn mức tăng tốc: RTF 0.40 → 0.21.
- CUDA graph làm CPU của codec gần như rảnh và tiếng đầu ra sớm hơn: 111 → 88 ms.
- Mảng vòng bớt việc cho GPU: 32 → 22 ms mỗi lần gọi.

Khi codec chạy không có graph, thời gian GPU đo được gồm cả lúc GPU chờ lệnh, nên không ghi. Số liệu:
[lite/results/ablation-20261015/](lite/results/ablation-20261015/).

## Bước 8. Bench so với SDK gốc

Đo đúng cách của giai đoạn 1 (xem [Phương pháp đo](PHASE1.md#phương-pháp-đo)), với SDK gốc và lite đã tối ưu,
profiler tắt ở cả hai bản. Kết quả là bảng ở đầu tài liệu.

**Máy đo:**

| Thông số | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12 GB, driver 570.195.03, giới hạn công suất 170 W |
| GPU UUID | `GPU-32502328-f35f-cbe2-bfa2-8cc40b1026c7` |
| Container | `0b29687edbe8` |
| Máy thuê | VKong `vk_b37c724fd417b0f8`, $0.078/giờ |
| Code | commit `60d316a` |
| Xung nhịp GPU | Không khóa được; 1837–1995 MHz trong lúc đo |
| Nhiệt độ GPU | 35–55 °C trong lúc đo |

Số liệu: [baseline/results/abba-20261016/](baseline/results/abba-20261016/) (SDK) và
[lite/results/abba-20261016/](lite/results/abba-20261016/) (bản tối ưu).

**Kiểm tra số trong bảng** (không cần GPU):

```bash
python3 experiments/phase1/analyze.py --abba abba-20261016
```

**Chạy lại.** Làm theo [các bước chạy lại của giai đoạn 1](PHASE1.md#chạy-lại-trên-vkong), chỉ thay bước 1 bằng:

```bash
mkdir -p /tmp/p2 && git archive 60d316a baseline lite | tar -x -C /tmp/p2
mv /tmp/p2/baseline /tmp/p2/sdk
cp baseline/results/abba-20261016/block-1-slots-16/vkong-deployed.yaml /tmp/p2/sdk/vkong.yaml
cp lite/results/abba-20261016/block-2-slots-16/vkong-deployed.yaml /tmp/p2/lite/vkong.yaml
```

Ở các bước sau, dùng `/tmp/p2/sdk` và `/tmp/p2/lite` thay cho `/tmp/p1/abba/sdk` và `/tmp/p1/abba/lite`. Tên App là
`vieneu-phase2`.

## Bước 9. Thử tải lâu và chất lượng

Mục tiêu: chạy đông người liên tục xem server có treo hay lỗi không, và âm thanh có bị méo không.

Server 32 ghế, cờ `BENCH_MODE=1`:

```bash
curl -X POST "$URL/stress?concurrency=32&duration=180"
curl "$URL/results/stress-32.json"
```

[stress.py](lite/stress.py) cho các client gửi câu liên tục, vào ra lệch giờ nhau. Sau đó nó dùng
[PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small) chép lại lời và tính tỉ lệ ký tự sai (CER). Nếu ghế
bị lẫn trạng thái thì CER lúc tải sẽ cao hơn hẳn lúc chạy riêng.

| Client | Thời gian | Request | Lỗi | TTFA P50 / P99 | RTF P50 / P99 | CER chạy riêng P50 / P90 | CER lúc tải P50 / P90 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3 phút | 1824 | 0 | 305 / 470 ms | 0.60 / 0.74 | 4.4% / 11.8% | 1.8% / 13.2% |
| 48 (hơn số ghế) | 2.5 phút | 1385 | 0 | 1833 / 2535 ms | 1.07 / 1.41 | 1.6% / 11.8% | 1.8% / 13.2% |

- Không lỗi nào. Ở 48 client, người dùng phải chờ ghế trống nên TTFA tăng.
- CER lúc tải ngang lúc chạy riêng.
- 112/1824 request (32 client) và 67/1385 (48 client) có âm thanh chạm ngưỡng ±1 và bị cắt. Chưa đo SDK theo cách này
  nên chưa biết đây là do model hay do tải.

Số liệu: [lite/results/stress-20261014/](lite/results/stress-20261014/), có file wav để nghe.

## Việc còn lại

- **Tiếng đầu khi đông người:** với 16 người vào cùng lúc, phần lớn thời gian nằm ở nạp câu (prefill chạy thường cho cả
  đợt) và hàng đợi. Có thể đưa prefill vào CUDA graph.
- **Codec giờ bị giới hạn bởi GPU:** mỗi lượt luôn tính đủ mọi ghế, kể cả ghế trống. Có thể thử `CODEC_DTYPE=bf16`, rồi
  đo lại chất lượng bằng nhận dạng giọng nói.

## Ghi chú khi làm với VKong

- **Giá.** Khi chạy không tương tác, `vkong run` đòi `--accept-price`; chốt mức giá trước. RTX 3060 12 GB khoảng
  $0.08–0.10/giờ. Trả máy ngay khi xong bằng `vkong app stop <tên app> --yes`.
- **Ghim loại card.** Ghi `gpu: RTX-3060-12GB`. Tên `RTX 3060` có thể thuê nhầm bản 8 GB hoặc 3060 Ti.
- **Kết nối hay đứt** (`Connection to the machine was lost`) trong khi máy vẫn chạy. Nối lại bằng `vkong attach`. Mọi
  phép đo đều chạy trên máy GPU, nên kết nối đứt không làm sai số đo.
- **Storage volume chặn `attach --update`.** Không cần giữ file sau khi trả máy thì đừng dùng volume; tải kết quả qua
  `/results/...`.
- **Thư mục project phải sạch** trước khi `vkong run`. Nếu lần cài đặt đầu chưa xong, `attach --update` báo
  `.vkong/init_cmd.sh: No such file`.
- **Không khóa được xung nhịp GPU** trên máy thuê, nên chỉ so các bản chạy trên cùng một máy.
- **Code của codec tải từ HuggingFace mỗi lần chạy**, không ghim phiên bản. Bật `CODEC_VERIFY=1` để chắc codec mới
  vẫn khớp codec gốc.
