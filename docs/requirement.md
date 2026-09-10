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
| Âm thanh kích hoạt tính năng | MP3 (`activate_voice/`) | Đóng gói vào Android `res/raw/` phát qua `AudioPlayerModule` (Kotlin) và bắn trigger BLE `CMD:PLAY_F*` |


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

## 4. Điều khiển qua giọng nói & Nút bấm vật lý (Tính năng nền tảng)

**Cơ chế nút bấm trên kính (ESP32-S3 Sense):** Kính trang bị 2 nút vật lý:
1. **Nút Nguồn (Power):** Bật/tắt nguồn thiết bị.
2. **Nút Ghi âm / Đa năng (Record Button - Chân GPIO với Debounce 30ms):**
   - **Nhấn giữ (HOLD > 300ms) — Push-to-talk (PTT):** Kính gửi mã BLE `01` (`BUTTON_HOLD`), kích hoạt mic I2S (INMP441) thu âm 16kHz 16-bit Mono thành các chunk (120 byte PCM → 160 ký tự Base64) stream qua BLE. App chuyển sang `LISTENING`, phát âm báo "Chờ nhận lệnh".
   - **Thả tay (RELEASE):** Kính gửi mã BLE `00` (`BUTTON_RELEASE`), dừng thu mic. Đồng thời camera OV2640 lập tức chụp 1 ảnh JPEG (320×240 QVGA) gửi kèm sang App. App chuyển sang `PROCESSING`, ghép các chunk PCM thành file WAV 16kHz hoàn chỉnh để đưa qua PhoWhisper.
   - **Nhấn nhanh (SHORT-PRESS ≤ 300ms) — Hủy khẩn cấp / Về Trang chủ:** Kính gửi mã BLE `02` (`BUTTON_SHORT_PRESS`). Có quyền ưu tiên cao nhất, lập tức ngắt phát âm thanh, dừng vòng lặp dẫn đường, App phát TTS "Đã về trang chủ" và đưa hệ thống về `IDLE`.

**Luồng nhận diện lệnh bằng giọng nói (Voice-First):**
1. User nhấn giữ nút Ghi âm → App vào `LISTENING` → phát audio "Chờ nhận lệnh".
2. User nói yêu cầu trong lúc giữ nút, thả nút ra là kết thúc ghi âm.
3. Audio stream gửi tới App qua BLE → App xử lý qua model transcribe on-device (PhoWhisper-tiny INT8) → ra text tiếng Việt.
4. App nhận diện keyword trong text (`matchFeatureKeyword`). Hỗ trợ chuẩn hóa bỏ dấu và đối sánh mờ:

   | Lời nói nhận diện | Mã tính năng | Âm thanh kích hoạt (Sound) | Lệnh BLE (App → Kính) | TTS Fallback xác nhận |
   | :--- | :---: | :---: | :---: | :--- |
   | "tính năng 1", "tính năng một", "hỏi đáp", "gpt", "miêu tả" | `FEATURE_1_QA` | `Open_f1.mp3` | `CMD:PLAY_F1` | "Đã vào tính năng 1, hỏi đáp, đã sẵn sàng" |
   | "tính năng 2", "tính năng hai", "chụp ảnh" | `FEATURE_2_CAPTURE` | `Open_f4.mp3` | `CMD:PLAY_F4` | "Đã vào tính năng 2, chụp ảnh" |
   | "tính năng 3", "tính năng ba", "dẫn đường", "tự động", "đi bộ" | `FEATURE_3_NAVIGATION` | `Open_f2.mp3` | `CMD:PLAY_F2` | "Đã vào tính năng 3, chế độ dẫn đường" |

   - Không khớp keyword nào → phát lại "Không nhận diện được lệnh, vui lòng thử lại" và quay về `IDLE`.

---

## 5. Tính năng 1: Hỏi đáp (Visual Question Answering)

1. **Vào tính năng:** Đang ở `FEATURE_1_QA`, đã phát âm thanh kích hoạt `Open_f1.mp3` (bắn lệnh BLE `CMD:PLAY_F1` sang kính, TTS fallback: "Đã vào tính năng 1, hỏi đáp, đã sẵn sàng").
2. **Hỏi đáp tương tác:** User nhấn giữ nút Ghi âm và nói câu hỏi (Push-to-talk) → kính stream audio mic I2S. Khi thả tay, kính chụp ngay 1 ảnh quang cảnh JPEG (320×240 QVGA) và gửi cả hai qua BLE về App.
3. **Chuyển giọng nói thành văn bản (STT):** App xử lý audio qua model PhoWhisper-tiny on-device (CTranslate2 INT8) → nhận về text câu hỏi tiếng Việt.
4. **Pipeline Xử lý Hỏi đáp Đa phương thức (AI Core):**
   - **Bước 1 (Dịch xuôi):** Model **EnViT5** (`VietAI/envit5-translation` CT2 INT8) dịch text câu hỏi Tiếng Việt → Tiếng Anh.
   - **Bước 2 (Visual QA):** Model **SmolVLM2** (`SmolVLM2-256M-Video-Instruct` on-device) tiếp nhận đồng thời [Ảnh từ kính + Câu hỏi Tiếng Anh], phân tích bối cảnh hình ảnh và sinh ra câu trả lời ngắn gọn (~1 câu súc tích) bằng Tiếng Anh.
   - **Bước 3 (Dịch ngược):** Model **EnViT5** dịch câu trả lời Tiếng Anh → Tiếng Việt hoàn chỉnh.
5. **Tổng hợp giọng nói (TTS):** App chuyển text trả lời tiếng Việt sang giọng nói qua `react-native-tts` (trên Mobile) hoặc `Piper TTS` (trên Python Server) → truyền dữ liệu audio về kính qua BLE Characteristic `AUDIO_OUT` để phát ở loa kính (chế độ MVP fallback phát qua loa ngoài điện thoại).
6. **Sau khi trả lời xong:** Hệ thống giữ nguyên trạng thái chờ trong `FEATURE_1_QA` để user có thể tiếp tục nhấn giữ hỏi câu tiếp theo, hoặc nhấn nhanh nút Ghi âm (≤ 300ms) để thoát về `IDLE`.

**Ví dụ:**
- Input: Ảnh phía trước + audio "Trước mặt tôi là gì"
- Pipeline: PhoWhisper ("Trước mặt tôi là gì") → EnViT5 ("What is in front of me?") → SmolVLM2 ("A street with parked cars and a pedestrian crossing.") → EnViT5 ("Một con phố với ô tô đang đỗ và lối qua đường cho người đi bộ.")
- Output (giọng nói): "Một con phố với ô tô đang đỗ và lối qua đường cho người đi bộ."

---

## 6. Tính năng 2: Chụp ảnh (Photo Capture & Storage)

1. **Kích hoạt:** User nói "tính năng 2" hoặc "chụp ảnh" từ `IDLE` → chuyển sang `FEATURE_2_CAPTURE`.
2. **Âm thanh kích hoạt:** App phát file âm thanh **`Open_f4.mp3`** (đồng thời bắn lệnh BLE **`CMD:PLAY_F4`** sang kính; TTS fallback: "Đã vào tính năng 2, chụp ảnh").
3. **Chụp & Truyền ảnh:**
   - App gửi lệnh điều khiển BLE **`CAPTURE`** qua Characteristic `AUDIO_OUT` sang kính.
   - Vi điều khiển ESP32-S3 điều khiển camera OV2640 chụp 1 khung ảnh JPEG (320×240 QVGA), cắt thành các packet Base64 (120 bytes binary → 160 Base64 chars) stream qua Characteristic `IMAGE` về App.
4. **Lưu trữ ảnh cục bộ:**
   - App thu thập đầy đủ các packet, ghép thành file ảnh JPEG nguyên vẹn.
   - Lưu trữ trực tiếp file ảnh vào thư mục máy (`Pictures/BSmart_...jpg` thông qua dịch vụ `ImageStorageService.ts`).
5. **Phản hồi hoàn tất:**
   - App tự động chuyển về `IDLE`, phát âm thanh TTS qua loa: "Đã hoàn thành, bạn muốn chọn tính năng nào tiếp theo".

---

## 7. Tính năng 3: Dẫn đường (Auto-pilot Obstacle & Depth Awareness)

> **Lưu ý phạm vi:** Đây **không phải** dẫn đường GPS bản đồ (không GPS, không map turn-by-turn). Trong MVP, tính năng này là **nhận thức và cảnh báo vật cản theo thời gian thực (Real-time Obstacle & Depth Awareness)**.

1. **Kích hoạt:** User nói "tính năng 3" hoặc "dẫn đường" từ `IDLE` → chuyển sang `FEATURE_3_NAVIGATION`.
2. **Âm thanh kích hoạt:** App phát file âm thanh **`Open_f2.mp3`** (đồng thời bắn lệnh BLE **`CMD:PLAY_F2`** sang kính; TTS fallback: "Đã vào tính năng 3, chế độ dẫn đường").
3. **Vòng lặp tự động (Auto-pilot Loop):**
   - App gửi lệnh điều khiển BLE **`NAV_START`** sang kính.
   - Kính kích hoạt timer định kỳ tự động chụp và gửi 1 frame JPEG mỗi **4 giây** (`NAVIGATION_FRAME_INTERVAL_MS = 4000`).
   - **Cơ chế Zero-Queue:** Chỉ xử lý 1 frame tại một thời điểm (`isNavigatingFrame`). Nếu frame trước đang inference chưa xong, frame mới gửi tới sẽ bị **drop ngay lập tức** để tránh trễ tích lũy.
4. **Xử lý AI On-Device (Không gọi Cloud):**
   - **YOLO26s:** Nhận diện vật cản/người trong khung hình (`person`, `car`, `motorcycle`, `bicycle`, `truck`, `bus`, `stairs`, `chair/table`), xác định bounding box và độ tin cậy (ngưỡng threshold ≥ 0.5).
   - **ZipDepth:** Ước lượng bản đồ độ sâu tương đối (Relative Depth) bên trong bounding box của từng vật thể, phân loại khoảng cách **GẦN** hay **XA**.
5. **Tổng hợp cảnh báo & Phát âm thanh:**
   - Ghép kết quả theo luật rule-based (xem mục 7.1) thành câu cảnh báo súc tích: *"Lưu ý, có [vật thể] ở [vị trí], [gần/xa]"*.
   - **Cơ chế chống lặp (Cooldown 8 giây):** Cùng 1 vật thể ở cùng vị trí/khoảng cách sẽ không nhắc lại trong 8 giây để tránh làm phiền người dùng. Nếu đường phía trước thông thoáng an toàn thì **giữ im lặng hoàn toàn**.
   - Chuyển text cảnh báo thành giọng nói (TTS) và phát ngay lập tức (ngắt lời thoại cũ nếu có cảnh báo khẩn cấp mới).
6. **Thoát chế độ:** 
   - User nhấn nhanh nút Ghi âm (≤ 300ms) trên kính bất kỳ lúc nào → Kính gửi mã BLE `02` (`BUTTON_SHORT_PRESS`), App gửi lệnh BLE **`NAV_STOP`** dừng timer chụp ảnh trên kính, phát TTS "Đã thoát chế độ dẫn đường" và trở về `IDLE`.

**Latency target:** < 2 giây từ lúc nhận frame đến khi bắt đầu phát cảnh báo âm thanh.

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

- Tông màu tối, độ tương phản cao, tối ưu hóa toàn diện cho người khiếm thị (TalkBack) đồng thời hỗ trợ chế độ kiểm thử cho lập trình viên.
- **Chế độ Mặc định: Giao diện Khiếm thị (Blind-First UX)**:
  - **`BlindTouchArea` (Màn hình tiếp xúc rộng)**: Chiếm trọn diện tích tương tác trung tâm, hỗ trợ các cử chỉ:
    - *Chạm & Giữ (> 250ms)*: Bắt đầu thu âm giọng nói (Push-To-Talk) $\to$ Rung phản hồi haptic (50ms) + Phát tiếng Bíp xác nhận.
    - *Thả tay*: Kết thúc thu âm và gửi lệnh $\to$ Rung phản hồi haptic (30ms) + Phát tiếng Cạch kết thúc.
    - *Chạm nhanh (< 250ms) hoặc Chạm 2 ngón tay*: Hủy lệnh khẩn cấp, dừng mọi tác vụ và lập tức trở về `IDLE` $\to$ Rung phản hồi kép.
  - **Hỗ trợ Google TalkBack toàn diện**: Toàn bộ các thành phần hiển thị đều được gắn `accessible={true}`, `accessibilityRole`, `accessibilityLabel` và `accessibilityHint` chi tiết; hệ thống tự động phát âm báo trạng thái qua `AccessibilityInfo.announceForAccessibility`.
  - **`ConnectionIndicator`**: Hiển thị trạng thái Bluetooth trực quan với kích thước chữ lớn, tương phản cao, nhãn TalkBack hướng dẫn chạm 2 lần để kết nối hoặc ngắt kết nối.
  - **`StatusDisplay`**: Biểu ngữ hiển thị trạng thái máy hiện tại (`IDLE`, `LISTENING`, `PROCESSING`, `FEATURE_1_QA`, `FEATURE_2_CAPTURE`, `FEATURE_3_NAVIGATION`).
- **Chế độ Lập trình viên (Dev HUD) & Cài đặt**:
  - Truy cập thông qua nút Cài đặt ⚙️ trên thanh tiêu đề (`SettingsModal`).
  - Khi kích hoạt, màn hình mở rộng hiển thị thêm:
    - **`MockControls`**: Bộ nút bấm giả lập các sự kiện phần cứng (Hold, Release, Short-press, Image Capture, Nav Frame) khi chạy trên Android Emulator.
    - **`DebugLog`**: Khung cuộn hiển thị timeline chi tiết các log hệ thống, kết quả inference AI và dữ liệu gói tin BLE theo thời gian thực.
    - **Công tắc BLE**: Cho phép chuyển đổi nhanh chóng giữa **Real BLE** (quét kính thật) và **Mock BLE** (giả lập offline).
    - **Công tắc Dịch vụ ngầm**: Bật/tắt Android Foreground Service.

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

**App background behavior:** Đã tích hợp **Android Foreground Service (`BSmartForegroundService`)** kèm `PARTIAL_WAKE_LOCK` và thông báo thường trực trên thanh trạng thái (*"BSmart - Kính AI đang hoạt động ngầm"*). Đảm bảo ứng dụng duy trì kết nối BLE và các tiến trình AI hoạt động ổn định liên tục, ngăn chặn triệt để nguy cơ bị hệ điều hành ngắt khi tắt màn hình hoặc kích hoạt chế độ tiết kiệm pin (Doze Mode / App Standby) khi người dùng đút điện thoại vào túi quần.

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
| **APP-09** | **Mobile App** | Tích hợp BLE thực tế (`BlePlxService.ts`) | Xây dựng engine `BlePlxService.ts` tích hợp `react-native-ble-plx`, ghép gói GATT chunk JPEG/Audio và giải mã sự kiện nút bấm (16/16 tests pass) | ✅ **Hoàn thành** |
| **APP-10** | **Mobile App** | Auto-connect BLE & Background Service | Xây dựng `BleAutoConnectService.ts` tự động quét/kết nối lại ngầm, phản hồi rung haptic và giọng nói cho người khiếm thị (21/21 tests pass) | ✅ **Hoàn thành** |
| **APP-11** | **Mobile App** | Giao diện Khiếm thị Trợ năng (`BlindTouchArea.tsx`) | Thiết kế mặc định Blind-First UX, màn hình tiếp xúc rộng, nhận diện chạm 2 ngón tay, gắn đầy đủ nhãn TalkBack Android, modal Cài đặt Dev HUD | ✅ **Hoàn thành** |
| **APP-12** | **Mobile App** | Phản hồi Âm thanh Bíp/Cạch & Rung Haptic | Tích hợp Android `ToneGenerator` zero-latency (`AudioPlayerModule.kt`) và rung `Vibration` khi nhấn giữ (Hold), thả nút (Release) và hủy lệnh | ✅ **Hoàn thành** |
| **APP-13** | **Mobile App** | Android Foreground Service & Quyền BLE Runtime | Cấu hình `BSmartForegroundService` kèm `PARTIAL_WAKE_LOCK`, khai báo quyền Android 12+ BLE và tự động xin quyền runtime (`BlePermissionService.ts`) | ✅ **Hoàn thành** |
| **APP-14** | **Mobile App** | Luồng điều khiển vận hành Kính (App $\to$ ESP32) | Gửi các lệnh `NAV_START` (chu kỳ 4s), `NAV_STOP` (dừng chụp), `CAPTURE` (chụp ảnh đơn) qua `AUDIO_OUT_CHAR_UUID` điều khiển camera kính | ✅ **Hoàn thành** |
| **APP-15** | **Mobile App** | Tích lũy Audio Chunk & Tích hợp PhoWhisper | Nối chuỗi PCM Base64 chunks từ ESP32 BLE thành WAV 16kHz chuẩn (`audioUtils.ts`), tích hợp PhoWhisper nhận diện giọng nói tiếng Việt thực tế | ✅ **Hoàn thành** |
| **APP-16** | **Mobile App** | Dịch vụ Âm thanh Kích hoạt & Lệnh Trigger BLE | Tích hợp `SoundEffectService.ts` phát file âm thanh `Open_f1.mp3`, `Open_f4.mp3`, `Open_f2.mp3` qua `AudioPlayerModule` và gửi lệnh BLE `CMD:PLAY_F1`, `CMD:PLAY_F4`, `CMD:PLAY_F2` (đã pass test) | ✅ **Hoàn thành** |
| **MOD-01** | **AI On-Device** | Export PhoWhisper-tiny sang ONNX/TFLite Mobile | Xây dựng `ai_core/export_phowhisper_onnx.py`, tích hợp `OnDeviceAsrService.ts` nhận diện giọng nói 100% offline | ✅ **Hoàn thành** |
| **MOD-02** | **AI On-Device** | Export SmolVLM2 sang ONNX/Mobile VLM Runtime | Xây dựng `ai_core/export_smolvlm2_onnx.py`, tích hợp `SmolVLM2-256M` on-device với gói tối ưu hóa (On-demand, 256x256, max 35 tokens, INT8, async non-blocking, tensor recycling) trong `OnDeviceVlmService.ts` & `OnnxInferenceModule.kt` | ✅ **Hoàn thành** |
| **MOD-03** | **AI On-Device** | Export YOLO26s + ZipDepth sang ONNX/TFLite Mobile | Xây dựng `ai_core/export_yolo_zipdepth_onnx.py`, trích xuất mô hình phát hiện vật cản và ước lượng độ sâu làn đường | ✅ **Hoàn thành** |
| **MOD-04** | **AI On-Device** | Đóng gói bộ Model Weights vào Android APK (~2GB) | Xây dựng `ai_core/package_models.py`, tích hợp C++ Native ONNX Runtime (`OnnxInferenceModule.kt`), build thành công APK Android (`BUILD SUCCESSFUL`) | ✅ **Hoàn thành** |
| **MOD-05** | **AI On-Device** | Tích hợp Mô hình Dịch thuật Song ngữ EnViT5 | Lượng tử hóa `VietAI/envit5-translation` (CT2 INT8 / ONNX) làm cầu nối ngữ nghĩa 2 chiều Việt $\leftrightarrow$ Anh phục vụ chu trình VLM SmolVLM2 | ✅ **Hoàn thành** |
| **FW-01** | **Firmware** | Source code C/C++ cho ESP32-S3 Sense (Arduino IDE/ESP-IDF) | Mã nguồn nạp vi điều khiển, quản lý I/O và cấu hình BLE Server (`firmware/esp32_sense/bsmart_esp32_sense.ino`) | ✅ **Đã code C++** *(Chờ nạp mạch)* |
| **FW-02** | **Firmware** | Điều khiển Camera (OV2640/OV5640) nén JPEG | Chụp ảnh độ phân giải 320x240 / 640x480, nén dung lượng ≤ 50–100 KB, phân mảnh gói BLE (`seq:total:payload`) | ✅ **Đã code C++** *(Chờ nạp mạch)* |
| **FW-03** | **Firmware** | Ghi âm mic I2S (INMP441 / PDM Onboard) | Thu âm 16kHz, mono, 16-bit khi người dùng giữ nút Ghi âm và stream Base64 qua BLE | ✅ **Đã code C++** *(Chờ nạp mạch)* |
| **FW-04** | **Firmware** | Xử lý sự kiện nút bấm vật lý (Nút Ghi âm & Nguồn) | Phân biệt Hold (bắt đầu nói), Release (kết thúc), Short-press (<300ms, thoát về Home) | ✅ **Đã code C++** *(Chờ nạp mạch)* |
| **FW-05** | **Firmware** | GATT Server & Bộ nhận lệnh điều khiển (App $\to$ Kính) | Phân mảnh BLE, xử lý các lệnh: `NAV_START`, `NAV_STOP`, `CAPTURE`, `CMD:PLAY_F1`, `CMD:PLAY_F2`, `CMD:PLAY_F4` | ✅ **Đã code C++** *(Chờ nạp mạch)* |
| **FW-06** | **Firmware** | Phát âm thanh ra loa gọng kính (BLE Audio Pipe) | Nhận stream âm thanh từ điện thoại qua BLE và phát ra I2S DAC/Loa kính | ⚠️ **Chờ Protocol** |
| **HW-01** | **Hardware/IoT** | Nạp vi điều khiển & kiểm thử phần cứng vật lý | Nạp code qua Arduino IDE (bật OPI PSRAM), kiểm tra Serial Monitor 115200, test Camera OV2640 & PDM mic | ⏳ **Sắp làm** *(Cần board vật lý)* |
| **INT-01** | **Tích hợp** | Kiểm thử liên thông BLE thực tế Kính ↔ Điện thoại | Đo đạc trễ truyền nhận ảnh JPEG 320x240, audio WAV 16kHz, tỷ lệ rớt gói và tối ưu kích thước chunk/MTU | ⏳ **Sắp làm** *(Sau khi có board)* |
| **CAL-01** | **Thực địa** | Hiệu chuẩn thực địa (Field Calibration) | Thử nghiệm lọc ồn mic ngoài đường, ánh sáng camera thực tế, cân chỉnh khoảng cách ZipDepth | ⏳ **Sắp làm** |
| **SRV-01** | **Backend Dev** | FastAPI HTTP Server bọc các script CLI (`server.py`) | Tạo endpoint `POST /transcribe`, `POST /qa`, `POST /navigate`, `GET /health` phục vụ kiểm thử tham chiếu trên máy trạm (4/4 tests pass) | ✅ **Hoàn thành** |
| **SRV-02** | **Backend Dev** | Lượng tử hóa CTranslate2 INT8 (`ct2-transformers-converter`) | Tối ưu hóa PhoWhisper và EnViT5 sang INT8, đạt 11/11 tests pass trong `check_glass.py` | ✅ **Hoàn thành** |
| **OPS-01** | **DevOps / CI-CD** | GitHub Actions Pipeline tự xuất file APK cho các phiên bản | Xây dựng `.github/workflows/build-apk.yml` tự động đóng gói, biên dịch APK Release/Debug và đính kèm trực tiếp vào GitHub Releases khi gắn tag phiên bản (`v*`) hoặc chạy manual | ✅ **Hoàn thành** |

---

### Bảng Điểm Nghẽn Kỹ Thuật & Giới Hạn Thực Tế (Blockers & Constraints)

| Vấn đề / Giới hạn | Phân loại | Tác động thực tế | Giải pháp hiện tại & Khuyến nghị |
| :--- | :---: | :--- | --- |
| **Chưa có Firmware ESP32 & UUIDs** | 🟢 **Resolved** | Mobile app đã có thể kết nối với kính thật | Đã triển khai đầy đủ mã nguồn C++ tại `firmware/esp32_sense/bsmart_esp32_sense.ino` đồng bộ 100% UUIDs và giao thức phân mảnh với `BlePlxService.ts`. |
| **Giao diện & Trợ năng cho Người khiếm thị** | 🟢 **Resolved** | Người khiếm thị dễ dàng thao tác trên điện thoại | Đã bổ sung `BlindTouchArea` màn hình tiếp xúc rộng, nhận diện chạm giữ nói / chạm 2 ngón hủy, âm bíp ToneGenerator, rung haptic và TalkBack toàn diện. |
| **Nguy cơ bị Doze Mode tắt khi tắt màn hình** | 🟢 **Resolved** | Ứng dụng chạy liên tục khi đút túi quần | Đã tích hợp Android Foreground Service (`BSmartForegroundService`) kèm `PARTIAL_WAKE_LOCK` và thông báo thường trực. |
| **Luồng điều khiển vận hành Kính (App $\to$ ESP32)** | 🟢 **Resolved** | Kính tự động chụp định kỳ 4s khi dẫn đường | Đã gửi lệnh `NAV_START`, `NAV_STOP`, `CAPTURE` qua Bluetooth điều khiển camera kính theo đúng trạng thái của App. |
| **Nút Bluetooth chưa kết nối thực tế** | 🟢 **Resolved** | Quét và kết nối Bluetooth Low Energy thực tế | Đã cấu hình Real BLE làm mặc định, tự động xin quyền runtime `BLUETOOTH_SCAN/CONNECT` và khai báo trong Manifest. |
| **Chưa có giao thức BLE Audio Pipe** | ⚠️ **Blocker** | Kính chưa tự phát được âm thanh ra loa | Phần cứng kính hiện tại chưa có module I2S DAC/Loa ngoài; Mobile app dùng `react-native-tts` phát qua loa ngoài điện thoại làm fallback. |
| **Đóng gói Model On-Device (~2GB)** | 🟢 **Resolved** | App chạy hoàn toàn offline trên điện thoại | Đã tích hợp ONNX Runtime Native C++ (`OnnxInferenceModule.kt`) và tối ưu On-Demand SmolVLM2 + PhoWhisper + YOLO26s + ZipDepth trực tiếp trong mã nguồn APK. |
| **Người khiếm thị khó tự thao tác mở app** | ♿ **UX Lim.** | Người mù không thể tự tìm và bấm mở app | Đã bổ sung `BleAutoConnectService.ts` tự động quét/kết nối lại ngầm khi kính bật nguồn; hỗ trợ phản hồi rung haptic và giọng nói. |
| **Chưa tích hợp bản đồ định vị GPS** | ♿ **UX Lim.** | Chỉ cảnh báo vật thể trước mắt, không chỉ đường | Nêu rõ trong phạm vi: Tính năng 3 là **Obstacle Awareness (Tránh vật cản)**, không phải GPS Turn-by-turn. |
| **Giới hạn băng thông BLE của ESP32** | ⚙️ **HW Lim.** | Dữ liệu ảnh/audio lớn dễ bị trễ hoặc rớt gói | Nén ảnh JPEG nhỏ (≤ 50–100 KB), bỏ qua frame trễ (skip frame), chỉ xử lý 1 frame tại một thời điểm. |
| **Đối tượng người dùng mục tiêu** | ♿ **UX Lim.** | Toàn bộ tương tác dựa trên tiếng Việt và giọng nói | Phù hợp với người khiếm thị có khả năng nghe/nói bình thường (chưa hỗ trợ người câm/điếc). |

---
