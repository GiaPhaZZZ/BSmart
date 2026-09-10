import serial
import time
import os
from pathlib import Path

ser = serial.Serial('COM7', 115200, timeout=5)
time.sleep(1)
ser.reset_input_buffer()

print("Sending 'c' to trigger camera capture...")
ser.write(b'c')

# Wait for START_JPEG:<len>
out_dir = Path("test/photo")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "camera_captured.jpg"

start_time = time.time()
found_start = False
jpeg_len = 0
jpeg_data = bytearray()

while time.time() - start_time < 6:
    line = ser.readline()
    if not line:
        continue
    text = line.decode('utf-8', errors='ignore').strip()
    print(f"[ESP32] {text}")
    if text.startswith("START_JPEG:"):
        jpeg_len = int(text.split(":")[1])
        print(f"Reading {jpeg_len} bytes of JPEG data...")
        jpeg_data = ser.read(jpeg_len)
        found_start = True
        break

ser.close()

if found_start and len(jpeg_data) > 0:
    with open(out_file, "wb") as f:
        f.write(jpeg_data)
    print(f"\n[SUCCESS] Captured image saved to: {out_file} ({len(jpeg_data)} bytes)")
else:
    print("\n[ERROR] Did not receive JPEG data")
