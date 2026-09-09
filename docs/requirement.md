# Requirement: App Điều Khiển Kính AI Hỗ Trợ Người Khiếm Thị

### Bản Demo

---

## 1. Tổng quan

| Mục | Nội dung |
| --- | --- |
| Tên dự án | Kính AI thông minh hỗ trợ dẫn đường người khiếm thị (BSmart) |
| Deadline | Trước 15/9 (MVP Demo) |
| Phần cứng | ESP32-S3 Sense, thiết kế dạng Omi glasses (mic + camera + loa, **không có màn hình**), 2 nút vật lý: nút **Nguồn** và nút **Ghi âm** |
| Ngôn ngữ | Ưu tiên tiếng Việt (`vi-VN`) |
| Nền tảng App | React Native 0.87.1, React 19, TypeScript |
| Thư viện chính | `react-native-ble-plx` (BLE), `react-native-tts` (TTS tiếng Việt) |

**Lưu ý quan trọng về âm thanh:** Vì user là người khiếm thị và kính không có màn hình, mọi phản hồi cho user phải là **giọng nói**, đích đến cuối cùng là phát ra ở **loa của kính** — không phải text hiển thị, và không phát ở loa điện thoại. Cụ thể theo thiết kế hoàn chỉnh:

1. Chuyển text đó thành audio bằng TTS.
2. Truyền file/luồng audio đó tới kính qua Bluetooth BLE.
3. Kính nhận và phát audio ra loa.

*Ghi chú triển khai MVP hiện tại:* Do giao thức stream audio BLE của firmware ESP32-S3 chưa sẵn sàng, app hiện tích hợp `react-native-tts` phát trực tiếp qua loa điện thoại làm fallback để đảm bảo toàn bộ luồng hội thoại và cảnh báo vật cản hoạt động trơn tru trong quá trình demo và test offline. Text trên màn hình app chỉ phục vụ dev debug và theo dõi trạng thái.

---

## 2. Kiến trúc hệ thống

```text
[Kính ESP32-S3 Sense] <===== BLE (GATT) =====> [App Điện Thoại Android (Offline On-Device)]
(Camera, Mic, Loa,                              ├── Điều phối trạng thái (State Machine), BLE Service
 Nút Ghi âm / Nguồn)                            ├── TTS Tiếng Việt (`react-native-tts`)
                                                └── Bộ Model AI Nhúng Trực Tiếp Trong App (~2GB):
                                                    ├── Transcribe: PhoWhisper (Nhận diện giọng nói)
                                                    ├── Visual QA: SmolVLM2 (Hỏi đáp hình ảnh)
                                                    └── Obstacle & Depth: YOLO26s + ZipDepth (Dẫn đường)
```

**Thay đổi kiến trúc quan trọng (Theo Ghi chú kỹ thuật):**
Toàn bộ các mô hình AI được chuyển trực tiếp vào trong App điện thoại Android để chạy **On-Device (Offline 100%)**, không phụ thuộc vào kết nối mạng, máy chủ Cloud hay máy chủ Python ngoài. App Android đảm nhận vai trò **tất cả trong một (All-in-One: Frontend UI + Backend AI Engine chạy ngầm bằng CPU/NPU điện thoại)**. Dung lượng file cài đặt ứng dụng (APK) dự kiến sẽ nặng khoảng **~2GB** do đóng gói kèm trọng số các mô hình AI.

- **Kính ESP32-S3:** Đóng vai trò thiết bị ngoại vi thu nhận I/O (chụp ảnh, ghi âm mic I2S, phát âm thanh ra loa gọng kính, gửi sự kiện nút bấm vật lý qua kết nối Bluetooth BLE). **Không xử lý AI.**
- **App điện thoại Android (All-in-One):** Trung tâm xử lý duy nhất — nhận dữ liệu từ kính qua Bluetooth BLE, chạy toàn bộ các model AI on-device ngầm bằng CPU điện thoại (PhoWhisper, SmolVLM2, YOLO26s, ZipDepth), quản lý state machine, tổng hợp giọng nói tiếng Việt (TTS) và truyền ngược âm thanh về loa kính.
- **Các file Python Script (`Function0`, `Function1`, `Function2`, `download_models.py`):** Đóng vai trò là **Mã nguồn tham chiếu (Reference Prototypes)** trên máy tính để kiểm thử thuật toán và làm công cụ export mô hình sang định dạng ONNX/TFLite/ExecuTorch cho điện thoại, KHÔNG phải là web server vận hành ứng dụng.
- **Cloud / Python Web Server:** **Không sử dụng trong kiến trúc vận hành chính thức.** Hệ thống vận hành hoàn toàn độc lập, offline 100% trên điện thoại.

### 2.1 Giao thức Bluetooth

- Dùng **BLE (GATT)**: ESP32 là **GATT Server**, App Android là **GATT Client**.
- Thư viện mobile app: `react-native-ble-plx` (React Native 0.87.1, React 19, TypeScript).
- Kiến trúc lớp trừu tượng `IBleService`:
  - `MockBleService`: Giả lập toàn bộ chu trình truyền ảnh, audio, sự kiện nút bấm phục vụ phát triển, test trên Android Emulator và chạy test tự động.
  - `BlePlxService`: Tích hợp BLE thực tế qua `react-native-ble-plx`, kết nối tới ESP32-S3, tự động chuyển đổi qua cờ cấu hình `USE_MOCK_BLE`.
- Dùng packet/chunk đơn giản với cấu trúc `type + length + sequence + payload`. Ảnh/audio được chia chunk và ghép lại ở App.
- Cần truyền được 2 chiều:
  - **Kính → App:** ảnh nén (JPEG), audio ghi âm, tín hiệu nút bấm (Record Hold, Record Release, Short Press).
  - **App → Kính:** audio phản hồi để phát ra loa kính (đang chờ firmware định nghĩa protocol; hiện tại app fallback phát qua loa điện thoại bằng `react-native-tts`).
- Lưu ý: Băng thông BLE của ESP32 còn giới hạn với các tệp ảnh và âm thanh lớn, do đó ảnh cần được nén JPEG độ phân giải phù hợp (320×240 hoặc 640×480, ≤ 50–100 KB).

### 2.2 Định dạng dữ liệu

| Loại dữ liệu | Định dạng | Ghi chú |
| --- | --- | --- |
| Ảnh camera | JPEG | Khuyến nghị ≤ 50–100 KB/ảnh để tránh nghẽn băng thông BLE |
| Audio ghi âm (kính → app) | PCM/WAV, 16 kHz, mono, 16-bit | Đầu vào cho model transcribe on-device (PhoWhisper) |
| Audio TTS (app → kính) | PCM 16kHz / Opus | Mục tiêu truyền về loa kính; hiện app dùng TTS loa điện thoại làm fallback |
| TTS engine | `react-native-tts` | Giọng tiếng Việt (`vi-VN`) của Android TTS engine, có hỗ trợ cancel và event listener |

### 2.3 Kiến trúc xử lý On-Device & Lưu trữ

- **Xử lý AI On-Device (Full Offline):**
  - Toàn bộ pipeline AI được đóng gói vào App Android, chạy thông qua runtime mobile tối ưu (ONNX Runtime Mobile, CTranslate2 INT8, TFLite hoặc ExecuTorch).
  - Trọng số model được lượng tử hóa (INT8 / FP16) để vừa vặn trong dung lượng bộ nhớ điện thoại (tổng dung lượng APK kèm model ~2GB).
  - *PhoWhisper-tiny*: Nhận diện giọng nói offline (tính năng 0 & 1).
  - *SmolVLM2*: Mô hình thị giác - ngôn ngữ nhỏ gọn, trả lời câu hỏi về hình ảnh offline (tính năng 1).
  - *YOLO26s + ZipDepth*: Nhận diện vật thể và ước lượng độ sâu tương đối theo thời gian thực (tính năng 3).
- **Lưu trữ:**
  - Ảnh chụp ở tính năng 2: lưu local trên bộ nhớ điện thoại (Gallery / Internal Storage).
  - Dữ liệu âm thanh và frame dẫn đường: xử lý trực tiếp trong RAM / bộ nhớ đệm tạm thời, giải phóng ngay sau khi inference hoàn tất.
  - Không cần database, không cần tài khoản người dùng, không cần đồng bộ mạng.
  - Log: lưu tạm trong React state của phiên chạy app, hiển thị trực quan trên màn hình qua `DebugLog`.

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
3. Audio gửi tới app qua Bluetooth → app xử lý qua model transcribe on-device (PhoWhisper) → ra text.
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
3. App xử lý audio qua model transcribe on-device (PhoWhisper) → ra text câu hỏi.
4. App đưa ảnh + text câu hỏi vào model SmolVLM2 on-device → nhận về text mô tả/trả lời.
5. App chuyển text trả lời sang giọng nói (TTS) → truyền audio về kính → phát ở loa kính.
6. Sau khi trả lời xong, quay về trạng thái chờ trong tính năng 1 (user có thể giữ nút hỏi tiếp) hoặc quay `IDLE` nếu nhấn nhanh nút Ghi âm.

**Ví dụ:**
- Input: ảnh + audio "Trước mặt tôi là gì"
- Output (giọng nói): "Trước mặt bạn là khung cảnh thành phố về đêm với đường xe chạy"

---

## 6. Tính năng 2: Chụp ảnh

1. User nói "tính năng 2" → vào `FEATURE_2_CAPTURE`, app phát audio xác nhận "Đã vào tính năng 2, chụp ảnh" → kính chụp ảnh → gửi ảnh tới app qua Bluetooth.
2. App lưu ảnh vào bộ nhớ điện thoại (local storage / thư viện ảnh).
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

- Tông màu tối, theme robot/high-tech / sci-fi HUD (sử dụng các mã màu `#0d1117`, `#00e5ff`, `#161b22`, v.v.).
- Vì thao tác chính là bằng giọng nói và nút bấm trên kính, UI trên điện thoại phục vụ giám sát, debug và kiểm thử:
  - **`StatusDisplay`**: Hiển thị trạng thái máy hiện tại (`IDLE`, `LISTENING`, `PROCESSING`, `FEATURE_1_QA`, `FEATURE_2_CAPTURE`, `FEATURE_3_NAVIGATION`) cùng text thông báo âm thanh mới nhất.
  - **`ConnectionIndicator`**: Hiển thị trạng thái kết nối Bluetooth (`Connected`, `Disconnected`, `Connecting`), kèm nút Reconnect thủ công.
  - **`DebugLog`**: Khung log dạng cuộn, hiển thị timeline các câu hỏi, phản hồi AI, cảnh báo vật cản và sự kiện hệ thống theo thời gian thực.
  - **`MockControls`**: Bảng điều khiển giả lập dành cho dev và demo khi không có kính vật lý:
    - Nút giả lập thao tác: Giữ nút Ghi âm (Hold Record), Thả nút (Release Record), Nhấn nhanh (Short Press - Thoát về Home).
    - Nút giả lập tính năng: Trigger Photo Capture, Trigger Nav Frame, Toggle chế độ Mock/Real BLE.

---

## 9. Xử lý lỗi & các trường hợp biên (Edge cases)

| Tình huống | Xử lý |
| --- | --- |
| Bluetooth disconnect | Dừng operation hiện tại, App về `IDLE`; sau khi reconnect thì cho phép sử dụng lại (không xử lý reconnect nâng cao — xem mục Out of scope) |
| Cloud API lỗi/timeout | Không crash App. Phát: "Không thể xử lý yêu cầu, vui lòng thử lại." Sau đó về state phù hợp |
| YOLO/ZipDepth inference lỗi | Bỏ frame đó, ghi log lỗi và tiếp tục frame tiếp theo |
| Đang phát audio mà có cảnh báo mới (Feature 3) | Không queue dài. Stop audio cũ và phát cảnh báo mới nhất |
| Short press trong khi đang xử lý | Short press có priority cao hơn recording/navigation, phải cancel operation hiện tại nếu có thể |

**Nguyên tắc audio output:** Tất cả system response và AI response mục tiêu thiết kế phải phát qua **loa kính**, không phát qua speaker điện thoại. Trong giai đoạn phát triển hiện tại khi giao thức stream BLE audio của kính chưa hoàn thiện, app sử dụng loa ngoài điện thoại qua `react-native-tts` làm giải pháp thay thế tạm thời.

**Hỗ trợ nhiều kính:** Không. MVP chỉ hỗ trợ **một kính tại một thời điểm**.

**App background behavior:** Có thể chạy background trong phạm vi cần thiết cho demo, nhưng không yêu cầu đảm bảo hoạt động khi Android kill process.

---

## 10. Cấu hình kỹ thuật

BLE UUID, API URL, timing, threshold và model configuration phải nằm trong config/constants tập trung, **không hard-code rải rác** trong code:
- `src/constants/bleUuids.ts`: Quản lý các UUID dịch vụ BLE và characteristic.
- `src/constants/apiConfig.ts`: Base URL và endpoint timeout.
- `src/constants/navigationRules.ts`: Ngưỡng nhận diện, ngưỡng khoảng cách, thời gian cooldown và chu kỳ frame.

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

1. **Model file YOLO26s / ZipDepth cụ thể** (định dạng ONNX/TFLite, input/output shape, label map) để nhúng trực tiếp vào mobile app.
2. **BLE UUID / protocol specification cụ thể từ firmware** (service UUID, characteristic UUID, chunk format truyền ảnh và audio).

---

## 13. Bảng Task Tổng Hợp & Tiến Độ Dự Án (Master Task Board)

### Bảng Điều Phối Task Toàn Diện (Task Matrix)

| Task ID | Phân hệ | Nhiệm vụ chi tiết | Định hướng kỹ thuật & Ghi chú | Trạng thái |
| :---: | :---: | --- | --- | :---: |
| **APP-01** | **Mobile App** | Khởi tạo dự án React Native 0.87.1, React 19, TypeScript | Build APK Android, cài đặt jest & test suite | ✅ **Hoàn thành** |
| **APP-02** | **Mobile App** | Cài đặt State Machine trung tâm (`useAppStateMachine.ts`) | Quản lý 6 trạng thái, ưu tiên short-press cancel khẩn cấp về Home | ✅ **Hoàn thành** |
| **APP-03** | **Mobile App** | Xây dựng Navigation Rule Engine (`NavigationEngine.ts`) | Phân vùng 33%/66%, relative depth, chọn top 2 vật cản, cooldown 8s (10/10 tests pass) | ✅ **Hoàn thành** |
| **APP-04** | **Mobile App** | Tích hợp Voice TTS tiếng Việt (`TtsService.ts`) | Thư viện `react-native-tts` (giọng `vi-VN`), hỗ trợ ngắt lời tức thì khi có cảnh báo mới | ✅ **Hoàn thành** |
| **APP-05** | **Mobile App** | Xây dựng Mock BLE Service (`MockBleService.ts`) | Giả lập 100% sự kiện nút bấm, truyền ảnh, audio và phát audio trên Emulator | ✅ **Hoàn thành** |
| **APP-06** | **Mobile App** | Xây dựng UI Dashboard giám sát (`MainScreen.tsx`) | Theme robot tối giản: StatusDisplay, ConnectionIndicator, DebugLog, MockControls | ✅ **Hoàn thành** |
| **APP-07** | **Mobile App** | API Client kết nối test (`ApiService.ts`) | Cung cấp endpoint `/transcribe` và `/qa` phục vụ giai đoạn dev trước khi có model on-device | ✅ **Hoàn thành** |
| **APP-08** | **Mobile App** | Lưu ảnh chụp tính năng 2 vào bộ nhớ máy | Tích hợp `ImageStorageService.ts` quản lý và lưu file ảnh chụp JPEG local kèm timestamp | ✅ **Hoàn thành** |
| **APP-09** | **Mobile App** | Tích hợp BLE thực tế (`BlePlxService.ts`) | Đã dựng cấu trúc; chờ firmware cung cấp Service/Characteristic UUIDs và chunk protocol | ⚠️ **Chờ Firmware** |
| **APP-10** | **Mobile App** | Auto-connect BLE & Background Service | Tự động kết nối lại khi kính bật nguồn mà người mù không cần nhìn màn hình | 📋 **Cần làm** |
| **MOD-01** | **AI On-Device** | Export PhoWhisper-tiny sang ONNX/TFLite Mobile | Chuyển đổi STT tiếng Việt để nhận diện giọng nói và câu hỏi hoàn toàn offline trong App | 📋 **Cần làm** |
| **MOD-02** | **AI On-Device** | Export SmolVLM2 sang ONNX/Mobile VLM Runtime | Tối ưu hóa mô hình hỏi đáp thị giác chạy cục bộ trên Android (offline) | 📋 **Cần làm** |
| **MOD-03** | **AI On-Device** | Export YOLO26s + ZipDepth sang ONNX/TFLite Mobile | Tối ưu hóa mô hình nhận diện vật thể và ước lượng độ sâu cho tính năng dẫn đường | 📋 **Cần làm** |
| **MOD-04** | **AI On-Device** | Đóng gói bộ Model Weights vào Android APK (~2GB) | Nhúng thư viện ONNX Runtime / ExecuTorch và nạp file model vào thư mục assets của app | ⚠️ **Chờ Model** |
| **FW-01** | **Firmware** | Source code C/C++ cho ESP32-S3 Sense (Arduino IDE/ESP-IDF) | Mã nguồn nạp vi điều khiển, quản lý I/O và cấu hình BLE Server | 📋 **Cần làm** |
| **FW-02** | **Firmware** | Điều khiển Camera (OV2640/OV5640) nén JPEG | Chụp ảnh độ phân giải 320x240 / 640x480, nén dung lượng ≤ 50–100 KB, chu kỳ 4s | 📋 **Cần làm** |
| **FW-03** | **Firmware** | Ghi âm mic I2S (INMP441) | Thu âm 16kHz, mono, 16-bit khi người dùng giữ nút Ghi âm | 📋 **Cần làm** |
| **FW-04** | **Firmware** | Xử lý sự kiện nút bấm vật lý (Nút Ghi âm & Nguồn) | Phân biệt Hold (bắt đầu nói), Release (kết thúc), Short-press (<300ms, thoát về Home) | 📋 **Cần làm** |
| **FW-05** | **Firmware** | Triển khai GATT Server & BLE Chunking Protocol | Chia nhỏ gói tin truyền ảnh/audio qua BLE, tối ưu MTU để tránh nghẽn băng thông | 📋 **Cần làm** |
| **FW-06** | **Firmware** | Phát âm thanh ra loa gọng kính (BLE Audio Pipe) | Nhận stream âm thanh từ điện thoại qua BLE và phát ra I2S DAC/Loa kính | ⚠️ **Chờ Protocol** |
| **SRV-01** | **Backend Dev** | FastAPI HTTP Server bọc các script CLI (`server.py`) | Tạo endpoint tạm thời `POST /transcribe` và `POST /qa` phục vụ kiểm thử tham chiếu trên máy tính trước khi nhúng model vào APK | ℹ️ **Tùy chọn Dev** |
| **SRV-02** | **Backend Dev** | Lượng tử hóa CTranslate2 INT8 (`ct2-transformers-converter`) | Tối ưu PhoWhisper và EnViT5 trên máy trạm test / phục vụ kiểm thử prototype | ⏳ **Đang làm** |

---

### Bảng Điểm Nghẽn Kỹ Thuật & Giới Hạn Thực Tế (Blockers & Constraints)

| Vấn đề / Giới hạn | Phân loại | Tác động thực tế | Giải pháp hiện tại & Khuyến nghị |
| :--- | :---: | :--- | --- |
| **Chưa có Firmware ESP32 & UUIDs** | ⚠️ **Blocker** | Mobile app chưa kết nối được kính thật | Dùng `MockBleService.ts` giả lập 100% sự kiện; team Firmware cần cung cấp mã nguồn và bảng UUIDs. |
| **Chưa có giao thức BLE Audio Pipe** | ⚠️ **Blocker** | Kính chưa tự phát được âm thanh ra loa | Mobile app tạm thời dùng `react-native-tts` phát qua loa ngoài điện thoại làm fallback. |
| **Chưa đóng gói Model On-Device (~2GB)** | ⚠️ **Blocker** | App chưa chạy độc lập offline hoàn toàn | Tạm thời dùng Local API (`ApiService.ts`) kết nối máy trạm; cần script export ONNX/TFLite để nhúng vào APK. |
| **Người khiếm thị khó tự thao tác mở app** | ♿ **UX Lim.** | Người mù không thể tự tìm và bấm mở app | Cần bổ sung Background Service tự khởi chạy và kết nối lại (Auto-reconnect) khi kính bật nguồn; thêm rung (Haptic feedback). |
| **Chưa tích hợp bản đồ định vị GPS** | ♿ **UX Lim.** | Chỉ cảnh báo vật thể trước mắt, không chỉ đường | Nêu rõ trong phạm vi: Tính năng 3 là **Obstacle Awareness (Tránh vật cản)**, không phải GPS Turn-by-turn. |
| **Giới hạn băng thông BLE của ESP32** | ⚙️ **HW Lim.** | Dữ liệu ảnh/audio lớn dễ bị trễ hoặc rớt gói | Nén ảnh JPEG nhỏ (≤ 50–100 KB), bỏ qua frame trễ (skip frame), chỉ xử lý 1 frame tại một thời điểm. |
| **Đối tượng người dùng mục tiêu** | ♿ **UX Lim.** | Toàn bộ tương tác dựa trên tiếng Việt và giọng nói | Phù hợp với người khiếm thị có khả năng nghe/nói bình thường (chưa hỗ trợ người câm/điếc). |

---
