# Requirement: App Điều Khiển Kính AI Hỗ Trợ Người Khiếm Thị

### Bản Demo

---

## 1. Tổng quan

| Mục | Nội dung |
| --- | --- |
| Tên dự án | Kính AI thông minh hỗ trợ dẫn đường người khiếm thị |
| Deadline | Trước 15/9 (8 ngày để làm) |
| Phần cứng | ESP32-S3 Sense, thiết kế dạng Omi glasses (mic + camera + loa, **không có màn hình**), 2 nút vật lý: nút **Nguồn** và nút **Ghi âm** |
| Ngôn ngữ | Ưu tiên tiếng Việt |
| Nền tảng App | TypeScript React Native |

**Lưu ý quan trọng:** Vì user là người khiếm thị và kính không có màn hình, mọi phản hồi cho user phải là **giọng nói**, phát ra ở **loa của kính** — không phải text hiển thị, và không phát ở loa điện thoại. Cụ thể, bất cứ khi nào app cần "thông báo" gì cho user, app phải:

1. Chuyển text đó thành audio bằng TTS.
2. Truyền file/luồng audio đó tới kính qua Bluetooth.
3. Kính phát audio ra loa.

Text chỉ tồn tại nội bộ giữa các bước xử lý, không phải thứ user "nhìn thấy".

---

## 2. Kiến trúc hệ thống

```text
[Kính ESP32-S3] <--Bluetooth--> [App điện thoại] <--Internet--> [Cloud]
                               |                    transcribe, SmolVLM
                               |
                               +--> On-device: YOLO26s + ZipDepth
                                    (tính năng 3, không cần internet)
```

Không cần quá lo phần cứng — miễn sao giữa app gửi và nhận tín hiệu qua Bluetooth với thiết bị pairing là ổn.

- **Kính:** chụp ảnh, ghi âm, phát audio ra loa, nhận tín hiệu nút bấm. **Không xử lý AI.** ESP32 chỉ đảm nhiệm I/O (camera, microphone, speaker, button); toàn bộ business logic nằm ở App.
- **App điện thoại:** trung tâm điều phối — nhận dữ liệu từ kính, gọi cloud (transcribe, SmolVLM) hoặc chạy model on-device (YOLO26s + ZipDepth cho tính năng 3), quản lý trạng thái, TTS hóa mọi phản hồi rồi truyền audio ngược về kính để kính phát ra loa (app **không** tự phát audio ở điện thoại).
- **Cloud:** chỉ chạy 2 model — transcribe (speech-to-text) và SmolVLM (hỏi đáp ảnh, tính năng 1). Mỗi model chỉ cần 1 API endpoint đơn giản, không cần chọn nhà cung cấp cụ thể.
- **On-device (điện thoại):** YOLO26s (nhận diện vật thể) + ZipDepth (ước lượng độ sâu) — dùng riêng cho tính năng 3, chạy hoàn toàn local để đảm bảo phản hồi thời gian thực, không phụ thuộc mạng. Ưu tiên **ONNX Runtime Mobile** hoặc **TFLite** tùy model thực tế; model được đóng gói cùng App.

### 2.1 Giao thức Bluetooth

- Dùng **BLE (GATT)**: ESP32 là **GATT Server**, App Android là **GATT Client**.
- Dùng packet/chunk đơn giản với cấu trúc `type + length + sequence + payload`. Ảnh/audio được chia chunk và ghép lại ở App.
- Cần truyền được 2 chiều:
  - **Kính → App:** ảnh nén, audio ghi âm, tín hiệu nút bấm.
  - **App → Kính:** audio phản hồi để phát ra loa kính.
- Không cần tối ưu phần cứng ở giai đoạn này.

### 2.2 Định dạng dữ liệu

| Loại dữ liệu | Định dạng | Ghi chú |
| --- | --- | --- |
| Ảnh camera | JPEG | Khuyến nghị ≤ 50–100 KB/ảnh để tránh truyền chậm/rớt kết nối |
| Audio ghi âm (kính → app) | PCM/WAV, 16 kHz, mono, 16-bit | |
| Audio TTS (app → kính) | Ưu tiên PCM cho MVP nếu băng thông BLE cho phép; nếu không ổn định thì chuyển sang **Opus** | |
| TTS engine | Android Text-to-Speech | |

### 2.3 Ghi chú Cloud & Lưu trữ (dành cho app dev)

- Model cloud (transcribe, SmolVLM) đều **tự host** trên cloud (tự tải model về và tự chạy trên server riêng — không gọi API bên thứ 3 như OpenAI/Google). App chỉ cần gọi API nội bộ do bên AI/infra cung cấp, không cần quan tâm server chạy trên máy gì.
- *(Tham khảo hạ tầng, không bắt buộc với app)*: đề xuất chạy trên **AWS EC2 g4dn.xlarge** (NVIDIA T4, 16GB VRAM, ~0.526 USD/giờ) — đủ cho demo với chi phí thấp, vì cloud giờ chỉ cần phục vụ 2 model (transcribe + SmolVLM). Nếu cần dư VRAM hơn, có thể dùng **g5.xlarge** (NVIDIA A10G, 24GB VRAM, ~0.916 USD/giờ). Lưu ý: dòng G5 không có tier "large", chỉ có từ "xlarge" trở lên.
- **API cloud app cần gọi** (bắt buộc phải biết để code):
  - `POST /transcribe` — input: file audio → output: `{ "text": "..." }`
  - `POST /qa` — input: ảnh (base64/multipart) + text câu hỏi → output: `{ "answer": "..." }` (model SmolVLM)
  - *(Định dạng field/tên endpoint chính xác do 2 bên thống nhất khi code; ở đây chỉ minh họa hình dạng request/response tối thiểu. Endpoint/field cụ thể có thể cấu hình bằng environment/build config.)*
  - **Cloud authentication:** không implement authentication phức tạp trong MVP nếu backend chưa yêu cầu.
- **Model on-device** (không phải API, chạy trực tiếp trong app): YOLO26s và ZipDepth được nhúng vào app dưới dạng model file (TFLite/ONNX tuỳ model thực tế) — không gửi request qua internet, không có "API" theo nghĩa network. Xem chi tiết luồng xử lý ở mục 7.
- **Lưu trữ:**
  - Ảnh chụp ở tính năng 2: lưu **local trên điện thoại** (demo không cần cloud storage).
  - Ảnh/audio dùng trong tính năng 1: chỉ gửi lên cloud để xử lý, xong thì xóa (kể cả trên cloud) — không lưu lại lâu dài, không cần cloud storage riêng.
  - Ảnh dùng trong tính năng 3: xử lý on-device, không gửi đi đâu cả, không cần lưu lại sau khi xử lý xong.
  - Không cần database, không cần tài khoản user cho demo.
  - Log: chỉ cần log trong phiên chạy App, **không cần persistence**.

---

## 3. Trạng thái hệ thống (State Machine)

```text
IDLE
 ↓ voice command
LISTENING
 ↓
PROCESSING
 ├── Feature 1 → FEATURE_1_QA
 ├── Feature 2 → FEATURE_2_CAPTURE → IDLE
 └── Feature 3 → FEATURE_3_NAVIGATION
                         ↓
                       IDLE
```

| Trạng thái | Mô tả |
| --- | --- |
| `IDLE` | Chờ nhận lệnh (mặc định / home) |
| `LISTENING` | Đang ghi âm lệnh từ user |
| `PROCESSING` | Đang xử lý (transcribe / gọi AI) |
| `FEATURE_1_QA` | Đang ở chế độ Hỏi đáp |
| `FEATURE_2_CAPTURE` | Đang chụp & lưu ảnh |
| `FEATURE_3_NAVIGATION` | Chế độ tự động dẫn đường (auto-pilot) |

Mọi phản hồi bằng giọng nói phát ra khi app chuyển trạng thái (ví dụ vào `LISTENING` → phát "Chờ nhận lệnh"). Error ở bất kỳ state nào → thông báo bằng voice → chuyển về state phù hợp.

---

## 4. Điều khiển qua giọng nói (Bắt buộc — tính năng nền tảng)

**Cơ chế ghi âm:** Push-to-talk — giữ nút Ghi âm khi nói, thả ra là kết thúc ghi âm (áp dụng cho mọi bước ghi âm trong toàn bộ app, kể cả tính năng 1).

**Luồng:**

1. User giữ nút Ghi âm trên kính → tín hiệu gửi qua app → app chuyển sang `LISTENING` → phát audio "Chờ nhận lệnh" qua loa kính.
2. User nói yêu cầu trong lúc giữ nút, thả nút ra là kết thúc ghi âm.
3. Audio gửi tới app qua Bluetooth → app gọi model transcribe (cloud) → ra text.
4. App nhận diện keyword trong text (mọi lựa chọn tính năng đều qua voice, không qua nút). Hỗ trợ normalization/fuzzy matching đơn giản:

   | Nói | Vào tính năng |
   | --- | --- |
   | "tính năng 1", "tính năng một", "GPT" | `FEATURE_1_QA` |
   | "tính năng 2", "tính năng hai" | `FEATURE_2_CAPTURE` |
   | "tính năng 3", "tính năng ba" | `FEATURE_3_NAVIGATION` |

   - Không khớp keyword nào → phát lại "Không nhận diện được lệnh, vui lòng thử lại" và quay về `IDLE`.

5. Khi vào tính năng, app luôn phát audio xác nhận nêu rõ tên tính năng vừa vào, ví dụ:
   - "Đã vào tính năng 1, hỏi đáp, đã sẵn sàng"
   - "Đã vào tính năng 2, chụp ảnh"
   - "Đã vào tính năng 3, chế độ dẫn đường"

**Thoát về trang chủ:** nhấn nhanh (short press, không giữ) nút Ghi âm bất kỳ lúc nào → thoát tính năng hiện tại, quay về `IDLE`, phát "Đã về trang chủ". (Nút Ghi âm dùng 2 kiểu thao tác: **giữ** = ra lệnh giọng nói; **nhấn nhanh** = về trang chủ.)

> Short press có priority cao hơn recording/navigation và phải cancel operation hiện tại nếu có thể.

**Nút Power:** short press → bật/tắt thiết bị. App không dùng Power để điều khiển feature. Nếu firmware thực tế có behavior khác thì firmware được ưu tiên.

---

## 5. Tính năng 1: Hỏi đáp

1. Vào: đã ở `FEATURE_1_QA`, đã phát audio xác nhận "Đã vào tính năng 1, hỏi đáp, đã sẵn sàng".
2. User giữ nút Ghi âm và nói câu hỏi (push-to-talk) → kính gửi audio + 1 ảnh chụp cùng lúc tới app qua Bluetooth.
3. App gửi audio tới model transcribe (cloud) → ra text.
4. App gửi ảnh + text câu hỏi tới cloud (model SmolVLM) → nhận về text mô tả/trả lời.
5. App chuyển text trả lời sang giọng nói (TTS) → truyền audio về kính → phát ở loa kính.
6. Sau khi trả lời xong, quay về trạng thái chờ trong tính năng 1 (user có thể giữ nút hỏi tiếp) hoặc quay `IDLE` nếu nhấn nhanh nút Ghi âm.

**Ví dụ:**
- Input: ảnh + audio "Trước mặt tôi là gì"
- Output (giọng nói): "Trước mặt bạn là khung cảnh thành phố về đêm với đường xe chạy"

---

## 6. Tính năng 2: Chụp ảnh

1. User nói "tính năng 2" → vào `FEATURE_2_CAPTURE`, app phát audio xác nhận "Đã vào tính năng 2, chụp ảnh" → kính chụp ảnh → gửi ảnh tới app qua Bluetooth.
2. App lưu ảnh (lưu local trên điện thoại là đủ cho demo, không cần đồng bộ cloud).
3. App tự động quay về `IDLE`, TTS hóa và phát qua loa kính: "Đã hoàn thành, bạn muốn chọn tính năng nào tiếp theo".

---

## 7. Tính năng 3: Dẫn đường (Auto-pilot) — Tính năng trọng tâm (USP)

> **Lưu ý phạm vi:** Đây **không phải** navigation thật sự (không GPS, không map, không route planning). Trong MVP, tính năng này là **real-time obstacle awareness**: nhận diện vật thể và cảnh báo vị trí/khoảng cách.

1. Vào chế độ qua voice ("tính năng 3") → `FEATURE_3_NAVIGATION`, app phát audio xác nhận "Đã vào tính năng 3, chế độ dẫn đường".
2. Cứ mỗi 4 giây, kính tự động chụp ảnh và gửi tới app qua Bluetooth.
   - Chu kỳ 4 giây tính từ lúc bắt đầu capture frame tiếp theo. Không tạo queue ảnh.
   - Chỉ xử lý **một frame tại một thời điểm** (không xử lý song song). Nếu frame trước chưa xong thì bỏ frame tiếp theo.
3. Ảnh được đưa vào 2 model chạy on-device trên điện thoại (không gọi cloud, để đảm bảo tốc độ phản hồi thời gian thực):
   - **YOLO26s** — nhận diện vật thể/người trong khung hình (class + vị trí bounding box).
   - **ZipDepth** — ước lượng độ sâu (khoảng cách tương đối) cho từng vùng ảnh, kết hợp với bounding box của YOLO để biết vật thể đó gần hay xa.
4. Kết quả 2 model được kết hợp thành text cảnh báo/chỉ dẫn (xem logic ở mục 7.1), sau đó chuyển thành giọng nói (TTS), truyền về kính và phát ở loa kính ngay khi có.
5. **Thoát chế độ:** nhấn nhanh nút Ghi âm → dừng vòng lặp chụp ảnh, quay về `IDLE`, phát "Đã thoát chế độ dẫn đường".
6. Nếu không có cảnh báo đặc biệt, **không phát audio** (tránh làm phiền).

**Latency target:** < 2 giây từ lúc nhận frame đến khi bắt đầu phát cảnh báo, nếu hardware đáp ứng.

*Bổ sung: Giao tiếp giữa ESP32-S3 Sense và Android*

Kính sử dụng **ESP32-S3 Sense**, do đó giao tiếp không dây giữa kính và ứng dụng Android được thực hiện thông qua **Bluetooth Low Energy (BLE)** theo mô hình **GATT**. Bluetooth Classic/SPP/RFCOMM không được sử dụng.

Trong phạm vi MVP/demo, BLE được sử dụng cho cả hai chiều:

* **Kính → Android:** truyền ảnh JPEG được camera chụp tự động mỗi 4 giây để Android xử lý bằng YOLO26s và ZipDepth.
* **Android → Kính:** truyền dữ liệu điều khiển và nội dung audio/TTS để kính phát qua loa.

Để đảm bảo tốc độ và đơn giản hóa việc triển khai demo, ảnh được chụp ở độ phân giải phù hợp với việc truyền qua BLE, dự kiến **320×240 hoặc 640×480 JPEG**. Ảnh được chia thành các chunk nhỏ để truyền qua GATT. Hệ thống không duy trì queue ảnh và chỉ xử lý một frame tại một thời điểm. Nếu frame trước chưa hoàn thành quá trình truyền hoặc xử lý thì frame tiếp theo sẽ được bỏ qua.

BLE chỉ được sử dụng ở mức **MVP/demo nhanh**, không yêu cầu truyền video hoặc stream dữ liệu liên tục. Kích thước ảnh, MTU, kích thước chunk và tốc độ truyền thực tế sẽ được điều chỉnh trong quá trình tích hợp firmware dựa trên kết quả thử nghiệm thực tế giữa ESP32-S3 Sense và thiết bị Android.*

### 7.1 Logic kết hợp output YOLO26s + ZipDepth thành câu cảnh báo

Vì 2 model chỉ trả về dữ liệu thô (bounding box + class từ YOLO, depth map từ ZipDepth), app cần một bước xử lý rule-based đơn giản để ra câu tiếng Việt tự nhiên.

**Detect object:** ưu tiên `person`, `car`, `bicycle`, `motorcycle`, `truck`, `bus`, `stairs`, `chair/table` hoặc obstacle tương tự nếu model hỗ trợ. Không tự train/fine-tune. **Confidence threshold** mặc định **0.5**, cho phép chỉnh trong debug/config.

**Xác định gần/xa (ZipDepth):** dùng relative depth, không coi là khoảng cách mét tuyệt đối. Lấy vùng depth tương ứng (trung bình depth trong bounding box) → chia thành `NEAR` / `FAR`, với threshold configurable để calibration thực tế.

**Xác định vị trí ngang (theo tọa độ x của bounding box):**

| Vị trí x trong khung hình | Vị trí |
| --- | --- |
| `< 33%` | Trái |
| `33% – 66%` | Giữa |
| `> 66%` | Phải |

**Ưu tiên khi có nhiều object:** tối đa **2 object** — ưu tiên object gần nhất và object nằm giữa màn hình; nếu cùng mức thì ưu tiên confidence cao hơn (tránh đọc quá nhiều thứ cùng lúc, gây rối cho user).

**Chống lặp:** cùng một `object + position + distance` có **cooldown khoảng 8 giây** để tránh đọc lặp liên tục.

**Câu mẫu:** "Lưu ý, có `{class}` ở `{vị trí}`, `{gần/xa}`" → ví dụ "Lưu ý, có người ở bên phải, gần" — có thể ghép nhiều cảnh báo trong 1 câu nếu cần.

Không cần model NLP/LLM để sinh câu — rule-based/template là đủ cho demo, ưu tiên tốc độ và độ ổn định hơn là câu văn tự nhiên hoàn hảo.

**Ví dụ:**
- Input: ảnh chụp mỗi 4s
- Output (giọng nói): "Lưu ý, có người bên phải; phía trước có bậc thang"

---

## 8. UI (điện thoại)

- Tông màu tối, theme robot/high-tech.
- Vì thao tác chính là bằng giọng nói và nút bấm trên kính, UI trên điện thoại cho demo chỉ cần:
  - Màn hình trạng thái hiện tại (`IDLE` / `Listening` / tên tính năng đang chạy).
  - Log dạng text các câu hỏi & trả lời gần nhất (để dev debug).
  - Trạng thái kết nối Bluetooth, hiển thị rõ **`Connected`**, **`Disconnected`**, **`Connecting`**.
- Không cần thiết kế phức tạp.

---

## 9. Xử lý lỗi & các trường hợp biên (Edge cases)

| Tình huống | Xử lý |
| --- | --- |
| Bluetooth disconnect | Dừng operation hiện tại, App về `IDLE`; sau khi reconnect thì cho phép sử dụng lại (không xử lý reconnect nâng cao — xem mục Out of scope) |
| Cloud API lỗi/timeout | Không crash App. Phát: "Không thể xử lý yêu cầu, vui lòng thử lại." Sau đó về state phù hợp |
| YOLO/ZipDepth inference lỗi | Bỏ frame đó, ghi log lỗi và tiếp tục frame tiếp theo |
| Đang phát audio mà có cảnh báo mới (Feature 3) | Không queue dài. Stop audio cũ và phát cảnh báo mới nhất |
| Short press trong khi đang xử lý | Short press có priority cao hơn recording/navigation, phải cancel operation hiện tại nếu có thể |

**Nguyên tắc audio output:** Tất cả system response và AI response phải phát qua **loa kính**, không phát qua speaker điện thoại.

**Hỗ trợ nhiều kính:** Không. MVP chỉ hỗ trợ **một kính tại một thời điểm**.

**App background behavior:** Có thể chạy background trong phạm vi cần thiết cho demo, nhưng không yêu cầu đảm bảo hoạt động khi Android kill process.

---

## 10. Cấu hình kỹ thuật

BLE UUID, API URL, timing, threshold và model configuration phải nằm trong config/constants tập trung, **không hard-code rải rác** trong code.

---

## 11. Ngoài phạm vi Demo (Out of scope)

- Tối ưu phần cứng, pin, độ bền kính.
- Xử lý lỗi mạng/mất kết nối Bluetooth nâng cao.
- Đa ngôn ngữ (chỉ tiếng Việt).
- Tài khoản người dùng, đồng bộ nhiều thiết bị.
- Fine-tune hoặc train lại YOLO26s/ZipDepth — dùng model pretrained có sẵn là đủ cho demo.
- Navigation thật sự (GPS, bản đồ, route planning) — Feature 3 chỉ là obstacle awareness.
- Authentication phức tạp cho cloud API.
- Đa kính (multi-device) — chỉ hỗ trợ 1 kính/thời điểm.
- Log persistence — chỉ log trong phiên chạy App.

---

## 12. Dependency cần artifact thực tế từ team liên quan (không tự suy đoán)

Hai phần sau **agent/dev không tự invent** — cần tạo **interface + mock** để phát triển song song, chờ artifact thật:

1. **Model file YOLO26s / ZipDepth cụ thể** (định dạng, input/output shape, label map).
2. **BLE UUID / protocol specification cụ thể từ firmware** (service UUID, characteristic UUID, chunk format thực tế).

---

*Tài liệu này hợp nhất nội dung từ `kinh_ai_requirement_v2.md` (requirement gốc) và `ai_glasses_mvp_decisions.md` (các quyết định làm rõ điểm mơ hồ cho MVP/demo).*