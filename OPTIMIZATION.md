# Giai đoạn 2: đo rồi mới sửa (14–15/10/2026)

Hết [giai đoạn 1](PHASE1.md), bản lite chạy ngang SDK gốc, và lần thử cho codec chạy song song cũng chỉ nhích một
chút. Giai đoạn này đổi cách làm: chưa sửa gì cả, đo trước đã. Dưới đây là nhật ký kể theo đúng thứ tự đã làm: đo cái
gì, từ số đo đoán ra điều gì, sửa ra sao, kiểm tra thế nào, và dùng VKong ở từng bước. Có ghi cả những lần đi sai
đường.

Kết quả cuối, đo trên cùng một máy RTX 3060 với 16 ghế (slot):

| | RTF 1 người | RTF 16 người | Tiếng đầu (TTFA), 1 người | TTFA, 16 người vào cùng lúc |
|---|---:|---:|---:|---:|
| Bản cũ | 0.402 | 0.538 | 124 ms | 294 ms |
| Bản tối ưu | **0.175** | **0.250** | **76 ms** | 300 ms |

RTF là thời gian xử lý chia cho độ dài âm thanh. Dưới 1 thì người nghe không bị ngắt. Model được giữ nguyên: trọng
số, kiến trúc, kiểu số và cách sampling đều không đổi. Âm thanh cũng không đổi: giải mã cùng một chuỗi mã thì tương
quan 0.9999993. Video tóm tắt: [docs/optimize.mp4](docs/optimize.mp4).

## Vòng làm việc

Mỗi bước dưới đây đều đi theo cùng một vòng:

1. **Đo** trên GPU thật: thuê máy qua VKong, bật profiler, cho chạy tải ngay trên máy GPU.
2. **Dự đoán** chỗ chậm từ số đo, rồi đọc code chỗ đó để biết vì sao.
3. **Sửa** mà không đổi phép tính của model.
4. **Kiểm tra âm thanh không đổi** trước khi đo tốc độ.
5. **Đo lại** trên cùng một máy, so với bản cũ, và đọc profile mới để tìm chỗ chậm kế tiếp.

Các cờ của `serve.py` dùng trong lúc đo (ghi vào dòng `start` của `lite/vkong.yaml`):

| Cờ | Dùng để |
|---|---|
| `PROFILE=1` | bật profiler: `/profile`, `/profile/trace`, frame anatomy trong log, watchdog |
| `BENCH_MODE=1` | mở `/benchmark?repeats=N` và `/stress` để chạy tải ngay trên máy GPU |
| `CODEC_VERIFY=1` | lúc khởi động, so codec mới với đường giải mã gốc của SDK |
| `MAX_STREAMS`, `OVERLAP_CODEC`, `CODEC_GRAPHS`, `CODEC_RING` | bật/tắt từng thay đổi để so |
| `WATERMARK=0` | tắt watermark như mọi lần đo trước |

## Bước 1. Gắn profiler (chưa cần GPU)

**Mục tiêu.** Trả lời được câu hỏi "thời gian đi đâu": phía CPU hay phía GPU, có bị nghẽn ở việc ra lệnh (launch
kernel) không, và mỗi request phải chờ ở chặng nào.

**Làm gì.** Viết [profiler.py](lite/vieneu_lite/profiler.py) và gắn vào scheduler, codec, frame và các thread nhận
request. Bật bằng `PROFILE=1`; khi tắt thì không tốn gì. Profiler cho biết:

- **Từng vùng code:** thời gian CPU (đã trừ các vùng con) và thời gian GPU (đo bằng hai CUDA event trên đúng stream
  của vùng đó).
- **Nhịp của server:** `tick.period` là thời gian tạo một mẩu cho mọi ghế, phải dưới 80 ms mới kịp real time.
  `tick.gap` là lúc GPU không chạy graph. `decode.backpressure` là lúc worker bị codec chặn. `gil.wait` cho biết các
  thread có tranh nhau GIL không.
- **Từng request:** thời gian tới tiếng đầu được tách thành chờ tới lượt, nạp câu, 2 mẩu đầu, chờ codec, codec, gửi đi.
- **Frame anatomy:** lúc khởi động, chạy một mẩu bằng eager và bằng graph để đếm kernel, in ra log.
- **Trace:** `torch.profiler` trên mọi thread, lưu file mở bằng [Perfetto](https://ui.perfetto.dev), kèm bảng số lần
  launch, thời gian launch, thời gian sync và thời gian kernel của từng vùng.
- **Watchdog:** có người dùng mà không có mẩu nào được tạo trong 30 giây thì in stack của mọi thread. GPU treo không
  để lại lỗi Python nào, nên đây là bằng chứng duy nhất.

Cách dùng khi server đang chạy:

```bash
curl -X POST "$SERVE_URL/profile/reset"             # bắt đầu một cửa sổ đo mới
# ... cho tải chạy ...
curl "$SERVE_URL/profile"                           # bảng tổng hợp (?format=json cho máy đọc)
curl -X POST "$SERVE_URL/profile/trace?seconds=2"   # trace 2 giây, lưu vào results/profile/
curl -O "$SERVE_URL/results/profile/<file>.json.gz" # tải trace về, kéo thả vào ui.perfetto.dev
```

**Hai cái bẫy đã thấy trước khi thuê máy,** bằng cách đọc source torch 2.8 và chạy thử trên CPU ở máy local:

- `torch.profiler` mặc định chỉ ghi thread gọi nó. Worker và codec chạy ở thread riêng nên trace sẽ trống. Phải bật
  `profile_all_threads=True`.
- Lite tự capture CUDA graph, mà torch chỉ tắt `TEARDOWN_CUPTI` cho graph của inductor. Không tắt thì lần trace thứ
  hai có thể crash, nên `PROFILE=1` tự đặt `TEARDOWN_CUPTI=0`.

## Bước 2. Đo bản cũ trên VKong

**Thuê máy.** Trong `lite/vkong.yaml` ghi `gpu: RTX-3060-12GB`, vì tên trần `RTX 3060` có thể thuê nhầm bản 8 GB hoặc
3060 Ti. Thêm `PROFILE=1` vào dòng `start`, rồi:

```bash
cd lite
vkong validate
vkong run --accept-price -y     # chỉ dùng --accept-price sau khi đã chốt mức giá
```

Đợi dòng `Workload running · http://127.0.0.1:PORT`, đó là tunnel tới server.

Lần thuê đầu (RTX 3060, $0.09/giờ, Nam Phi) có gắn thêm một storage volume để lấy trace về. Kết quả đầu tiên đến từ
frame anatomy trong log:

```
eager: 1965 GPU ops, host 49.0 ms to launch them, GPU span 49.0 ms
graph: 1 launch, host 0.04 ms, GPU 8.8 ms
```

Tức là graph của model sinh âm đã làm đúng việc của nó: bỏ được khoảng 40 ms ra lệnh mỗi mẩu. Máy này mất kết nối giữa
chừng. Ở máy thứ hai (Texas), volume lại chặn `attach --update` (VKong báo `stop the Storage writer before starting
another Run`), nên từ máy thứ ba bỏ volume, tải file qua tunnel `/results/...`.

**Đo 1 người dùng, không trace** (máy thứ hai, Texas):

```bash
SERVE_URL=http://127.0.0.1:PORT
curl -X POST $SERVE_URL/profile/reset
# cho 1 client gửi câu liên tục khoảng 15 giây, rồi:
curl $SERVE_URL/profile
```

Những dòng quyết định:

```
tick.period       p50  11.8 ms                    → một mẩu cho cả 16 ghế, nhanh gấp ~7 lần real time
codec.decode      host p50 182 ms | gpu p50 183 ms → mỗi lần gọi codec (4 mẩu = 320 ms âm thanh)
decode.backpressure    p50 129 ms                 → worker ngồi đợi codec
req.first_audio   167 ms = admit 47 + 2 mẩu đầu 19 + codec 103
```

**Đo có trace** để biết codec bận ở CPU hay GPU:

```bash
curl -X POST "$SERVE_URL/profile/trace?seconds=2"     # trong lúc đang có tải
curl -O $SERVE_URL/results/profile/<file>.json.gz      # mở bằng ui.perfetto.dev
```

Mỗi lần gọi codec: khoảng 3500 phép tính nhỏ, ~1750–2450 lần launch kernel, **54 ms kernel GPU** cho **182 ms CPU**,
khoảng 30% thời gian CPU là Python chạy giữa các phép. GPU của cả máy chỉ bận 20% thời gian.

**Dự đoán.** Model sinh âm không phải chỗ chậm. Codec mới là chỗ chậm, và nó bị giới hạn bởi **CPU ra lệnh**, không
phải bởi GPU tính: GPU có việc khoảng 54/182 ≈ 30% thời gian. Riêng codec đã ăn RTF ~0.57.

## Bước 3. Đọc code codec (chưa cần GPU)

Tải source `vieneu==3.8.3` từ PyPI và code của codec
([MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)) về máy local đọc.

**Codec nhận gì.** Mỗi mẩu 80 ms, model sinh 16 số (0–1023) cho mỗi người; mỗi lần gọi codec nhận 4 mẩu. Codec gộp
16 số thành một vector, rồi qua 4 tầng transformer (12 lớp), chia mỗi mẩu dần thành 4 → 8 → 16 → 32 bước thời gian,
cuối cùng ra 3840 mẫu × 2 kênh = 80 ms. Mỗi lớp chỉ nhìn lại một cửa sổ quá khứ 4–10 giây, và giữ key/value của cửa
sổ đó trong KV cache: khoảng 400 MB với 16 ghế.

**Ba chỗ làm codec chậm:**

1. `MossAudioTokenizerDecodeSession.step` chạy **từng mẩu một**: cả 12 lớp cho mẩu 1, rồi mới tới mẩu 2.
2. Sau mỗi mẩu, nó đọc độ dài âm thanh của từng người về CPU bằng `.item()`. Lần đọc đầu tiên trong mỗi mẩu bắt CPU
   đợi GPU làm xong, nên CPU không thể ra lệnh trước cho mẩu sau. Hai bên thay nhau chờ.
3. Attention streaming, ở mỗi lớp và mỗi lượt, **ghi lại toàn bộ cửa sổ KV 4 lần** (`cat` → cắt → `where` → `copy_`),
   dù chỉ thêm vài bước mới. Bốn mẩu làm lần lượt thành 16 lần mỗi lần gọi.

Ghi chú trong code upstream nói chế độ `use_cuda_graph` của codec "rẻ hơn 3 lần nhưng ra âm thanh sai". Đọc code thì
thấy lý do: `step` giữ các view trỏ vào buffer output tĩnh của graph qua từng mẩu, lần phát sau ghi đè lần trước.

**Dự đoán cách sửa:**

- Codec không phải mô hình tự hồi quy: các con số của cả 4 mẩu đã có sẵn từ model. Nó chỉ cần không nhìn tương lai
  (mask nhân quả), nên **4 mẩu tính trong một lượt** cho đúng kết quả như 4 lượt, giống prefill của LLM. Bỏ luôn
  `.item()` vì độ dài đã biết trước (4 × 80 ms).
- Sau khi gom, **CUDA graph** cho codec sẽ bỏ gần hết thời gian CPU ra lệnh. Phải chép output ra trước khi phát lần sau.
- Thay việc ghi lại cả cửa sổ bằng **KV cache dạng vòng**: mảng cố định dài cửa sổ + 4 mẩu, bước mới ghi đè ô của
  bước cũ nhất đã ra khỏi cửa sổ, thứ tự được tính lại từ số thứ tự bước khi làm attention.

## Bước 4. Viết `SlotCodec` và chứng minh âm thanh không đổi

[codec.py](lite/vieneu_lite/codec.py) thay đường giải mã của SDK: slot của codec trùng slot của scheduler, các người
dùng có cùng số mẩu mới đi chung một lượt, mỗi độ dài 1–4 mẩu có một CUDA graph, KV cache dạng vòng.

**Kiểm tra trước khi đo tốc độ.** Hàm `verify()` cho cùng một chuỗi mã ngẫu nhiên đi qua đường fp32 gốc của SDK và
qua `SlotCodec`: hai người dùng, dài ~13 giây để vòng quay hết, có người vào giữa chừng, có người rời đi, số mẩu lẫn
lộn. Chạy hai nơi:

- **Trên CPU ở máy local**, với code codec tải từ HuggingFace và trọng số ngẫu nhiên: vài giây, không tốn GPU, dùng để
  sửa logic vòng. Kết quả tương quan 0.9999999.
- **Trên GPU với trọng số thật**, bằng `CODEC_VERIFY=1` mỗi lần server khởi động. Kết quả ghi trong log và `/health`:
  tương quan **0.9999993**, sai lệch lớn nhất 0.0105 trên thang ±1.

## Bước 5. A/B trên cùng một máy, và lần GPU treo thật

Muốn so công bằng thì bản cũ và bản mới phải chạy trên **cùng một máy**, vì máy thuê không cho khóa xung nhịp GPU và
mỗi máy một khác. Bản cũ lấy đúng code đã commit, đặt ở một thư mục nháp, rồi đẩy lên máy đang thuê:

```bash
mkdir /tmp/lite-baseline
git archive HEAD lite | tar -x -C /tmp/lite-baseline --strip-components=1
# trong /tmp/lite-baseline/vkong.yaml: thêm BENCH_MODE=1 vào start, gpu: RTX-3060-12GB
vkong instances                                       # lấy <instance-id>
vkong attach <instance-id> -C /tmp/lite-baseline --update
```

Lần đó máy vừa thuê chưa cài xong thư viện (đồng bộ file hỏng giữa chừng vì có file trace rác trong `lite/`), nên phải
dùng `--setup` thay cho `--update`, và bỏ dòng `.vkong/` khỏi `.vkongignore` của thư mục nháp để script cài đặt được
gửi lên.

Đo bằng client chạy ngay trên máy GPU, để tunnel không lẫn vào số đo:

```bash
curl -X POST "$SERVE_URL/benchmark?concurrency=16&repeats=8"   # 2 lượt khởi động + 8 lượt đo
curl $SERVE_URL/benchmark                                       # đợi tới khi running=false
curl -O $SERVE_URL/results/bench-16.jsonl                       # từng request: TTFA, RTF
```

Rồi đẩy bản mới lên cùng máy, `cd lite && vkong attach <instance-id> --update`, và đo lại y hệt.

**GPU treo lúc khởi động.** Bản mới không lên: không lỗi, không log. Bật `faulthandler` cho in stack của mọi thread khi
khởi động quá 60 giây, đẩy lại, và thấy:

```
codec thread : ... audio.cpu()             ← đứng chờ GPU
worker thread: ... torch.cuda.Event.record ← đứng chờ GPU
```

Cả hai đều chờ GPU, nghĩa là GPU kẹt thật. **Dự đoán:** `torch.cuda.graph` mặc định capture mọi graph trên cùng một
stream, mà cuBLAS cấp vùng nháp (workspace) theo stream, nên graph của model và graph của codec dùng chung một vùng
nháp; khi hai graph chạy song song thì kẹt. **Sửa:** capture graph của codec trên stream riêng. Hết treo. Phần nguyên
nhân là suy luận từ việc bản sửa có tác dụng, chưa chứng minh trực tiếp. Từ đó có thêm **watchdog**: có người dùng mà
không có mẩu nào được tạo trong 30 giây thì in stack mọi thread.

**Một lần đi sai đường.** Sau đó đo ở 16 người, tunnel đứt và `/health` không trả lời; mình kết luận "overlap làm GPU
treo". Nhưng chạy lại với overlap tắt, tunnel vẫn đứt y như vậy trong khi máy vẫn `running`. Đó là tunnel của VKong,
không phải GPU. Bài học: chỉ tin stack dump của watchdog, đừng tin một tunnel đứt.

**Số đo A/B (16 ghế, cùng máy, trung vị 8 lượt).** Bản mới lúc này mới có gom + graph, chưa có vòng; riêng cột
"overlap bật" đo sau khi đã sửa thêm phần lead-in và số người nhận mỗi lượt (xem ngay dưới):

| Người dùng | TTFA cũ | TTFA mới, overlap tắt | TTFA mới, overlap bật | RTF cũ | RTF mới, overlap tắt | RTF mới, overlap bật |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120.5 | **81.8** | 87.1 | 0.402 | 0.220 | **0.208** |
| 4 | 189.9 | **154.4** | 178.4 | 0.442 | 0.254 | **0.250** |
| 16 | 310.0 | **273.6** | 344.9 | 0.533 | 0.334 | **0.307** |

**Profile mới cho thấy chỗ chậm kế tiếp:**

- Codec giờ bị giới hạn bởi GPU: CPU ~0.6 ms, GPU ~28 ms mỗi lần gọi. Vì vậy làm tiếp KV cache dạng vòng.
- Ngoài mốc 4 mẩu, mỗi khi có người mới thì code cũ giải mã luôn cả người cũ, tốn thêm một lượt codec. Sửa thành chỉ
  giải mã người mới.
- Khi 16 người vào cùng lúc, mỗi tick chỉ nhận 8 người, nên đợt đó phải nạp câu 2 lần. Nâng lên 16.

## Bước 6. Ổn định và chất lượng khi chạy batch lớn

Chạy 32 ghế, overlap bật. Dùng [stress.py](lite/stress.py) qua `/stress`, chạy ngay trên máy GPU: mỗi client gửi câu
liên tục, vào ra lệch giờ nhau để người dùng liên tục vào và rời batch.

```bash
curl -X POST "$SERVE_URL/stress?concurrency=32&duration=180"
curl $SERVE_URL/results/stress-32.json               # tóm tắt; stress-32.jsonl từng request; stress-32/*.wav
```

| Client | Thời gian | Request | Lỗi | TTFA p50 / p99 | RTF p50 / p99 |
|---:|---:|---:|---:|---:|---:|
| 32 | 3 phút | 1824 | 0 | 305 / 470 ms | 0.60 / 0.74 |
| 48 (vượt 32 ghế) | 2.5 phút | 1385 | 0 | 1833 / 2535 ms | 1.07 / 1.41 |

Không treo, không lỗi; watchdog không lần nào báo. Ở 48 client, người dùng phải chờ ghế trống nên TTFA tăng.

**Đo chất lượng thế nào.** Ý định đầu tiên là so dạng sóng ở chế độ greedy: một câu đọc riêng, cùng câu đó đọc lúc
tải nặng. Nhưng tương quan chỉ 0.08: model sinh âm chạy bf16, và prefill gom batch khác nhau cộng với vị trí trong
KV cache khác nhau làm `argmax` rẽ nhánh ngay từ vài mẩu đầu. Cách đo đó không dùng được. Thay bằng **ASR**:
`stress.py` cho [PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small) chép lại lời rồi tính tỉ lệ ký tự
sai (CER). Nếu ghế bị lẫn trạng thái thì audio sẽ méo và CER khi tải sẽ cao hơn hẳn khi chạy riêng.

| | CER chạy riêng p50 / p90 | CER khi tải p50 / p90 |
|---|---:|---:|
| 32 client | 4.4% / 11.8% | 1.8% / 13.2% |
| 48 client | 1.6% / 11.8% | 1.8% / 13.2% |

**Mẫu nghe so sánh.** Cùng seed trên một server đang chạy vẫn ra mã khác nhau (tương quan 0.09–0.22), vì con trỏ ghi
KV cache của model tăng dần qua các request. Cách làm: khởi động lại từng bản, gọi `/benchmark/seed?seed=42` trước
mỗi câu, đọc 5 câu lần lượt. Hai bản khi đó ở cùng trạng thái nên model sinh **cùng một chuỗi mã**, khác biệt chỉ còn
ở codec: độ dài khớp từng mẫu, tương quan 0.9999998, SNR ~64 dB. File ở [docs/samples/](docs/samples/), chẳng hạn
[original-1.wav](docs/samples/original-1.wav) và [optimized-1.wav](docs/samples/optimized-1.wav).

## Bước 7. Đo từng phần (ablation)

Câu hỏi còn lại: nhanh lên nhờ gom mẩu, nhờ CUDA graph hay nhờ vòng? Thuê một máy (RTX 3060, $0.08/giờ, Hàn Quốc),
giữ 16 ghế, overlap bật, và bật tắt từng thay đổi bằng cờ trong `vkong.yaml`. Mỗi cấu hình chỉ cần đổi dòng `start`
rồi `vkong attach <instance-id> --update` (khoảng 1 phút, không cài lại thư viện), sau đó chạy `/benchmark` như bước 5.

```
start: env PROFILE=1 BENCH_MODE=1 MAX_STREAMS=16 CODEC_GRAPHS=1 CODEC_RING=0 WATERMARK=0 OMP_NUM_THREADS=4 python -u serve.py
```

| Cấu hình | RTF 1 người | RTF 16 người | TTFA 1 người | TTFA 16 người | Codec mỗi lần gọi: CPU / GPU |
|---|---:|---:|---:|---:|---:|
| Bản cũ (HEAD) | 0.402 | 0.538 | 124 ms | 294 ms | — |
| Gom | 0.208 | 0.322 | 111 ms | 325 ms | 40 ms / —\* |
| Gom + vòng | 0.180 | 0.288 | 104 ms | 340 ms | 34 ms / —\* |
| Gom + graph | 0.209 | 0.306 | 88 ms | 338 ms | 1.3 ms / 32 ms |
| Gom + graph + vòng | **0.175** | **0.250** | **76 ms** | 300 ms | 1.2 ms / 22 ms |

\* Khi codec chạy eager, khoảng thời gian GPU đo bằng CUDA event gồm cả lúc GPU đứng chờ lệnh nên không ghi.
Số liệu: [lite/results/ablation-20261015/](lite/results/ablation-20261015/), tóm tắt ở `summary.json`.

- **Gom mẩu** cho phần lớn mức tăng tốc: RTF 0.40 → 0.21.
- **CUDA graph** làm CPU của codec gần như rảnh (40 → 1.3 ms). RTF gần như không đổi, vì codec đã chạy song song với
  model, nhưng tiếng đầu ra sớm hơn: 111 → 88 ms.
- **Vòng** bớt việc cho GPU (32 → 22 ms mỗi lần gọi), RTF giảm thêm 16–18%.
- **Chưa cải thiện:** khi 16 người vào cùng lúc, tiếng đầu vẫn ~300 ms. Profile của bản tối ưu: 263 ms phía server =
  chờ tới lượt 68 + nạp câu 129 + 2 mẩu đầu 39 + codec 19.

## Việc còn lại

- **Tiếng đầu khi đông người:** nạp câu (embedding + prefill eager cho cả đợt) và hàng đợi chiếm phần lớn. Có thể đưa
  prefill vào CUDA graph theo nhóm độ dài, và ưu tiên lượt codec đầu tiên của người mới.
- **Codec giờ bị giới hạn bởi GPU:** mỗi lượt luôn tính đủ mọi ghế, kể cả ghế trống. Có thể thử `CODEC_DTYPE=bf16`,
  nhưng phải đo lại chất lượng bằng ASR.

## Ghi chú khi làm với VKong

- **Giá.** Ở chế độ không tương tác (agent), `vkong run` không in giá mà đòi `--accept-price`; chốt mức giá với người
  dùng trước. RTX 3060 12 GB khoảng $0.08–0.10/giờ. Trả máy ngay khi xong:
  `vkong app stop <app-id> --yes` (có nhiều app trùng tên thì dùng ID trong `vkong app`).
- **Tunnel hay đứt** (`Connection to the machine was lost`) trong khi máy vẫn `running`. Chạy lại `vkong attach <id>`
  để mở tunnel mới; mọi phép đo đều chạy trên máy GPU, qua tunnel chỉ hỏi trạng thái và tải file.
- **Storage volume chặn `attach --update`** (đã gửi feedback fb-133). Không cần giữ file sau khi trả máy thì đừng dùng
  volume; tải kết quả qua `/results/...`.
- **Lần cài đặt đầu chưa xong** (ví dụ đồng bộ file hỏng giữa chừng vì có file rác trong thư mục project) thì
  `attach --update/--setup` báo `.vkong/init_cmd.sh: No such file`, vì `.vkongignore` loại `.vkong/`. Thư mục project
  phải sạch trước khi `vkong run`.
- **Không khóa được xung nhịp GPU** trên máy thuê (`nvidia-smi -lgc` bị từ chối), nên chỉ so các bản trên cùng một máy.
