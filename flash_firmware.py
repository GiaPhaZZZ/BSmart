import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def find_esp32_port():
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = (p.description or '').lower()
            hwid = (p.hwid or '').lower()
            if any(k in desc or k in hwid for k in ['ch340', 'ch343', 'cp210', 'ftdi', 'usb-serial', 'uart', '1a86', '10c4', '303a']):
                return p.device
        # Fallback to first non-bluetooth port
        for p in ports:
            if 'bluetooth' not in (p.description or '').lower():
                return p.device
    except ImportError:
        pass
    return None

def main():
    print("=" * 60)
    print("      BSmart — One-Click ESP32-S3 Firmware Flasher")
    print("=" * 60)

    # 1. Check/Install esptool and pyserial
    try:
        import esptool
    except ImportError:
        print("[!] esptool chua duoc cai dat. Dang cai dat tu dong qua pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "esptool", "pyserial"])
        import esptool

    # 2. Find COM Port
    port = find_esp32_port()
    if not port:
        print("\n[!] Khong tim thay mach ESP32 qua cong USB.")
        print("    Vui long kiem tra day cap USB va cam lai vao may tinh.")
        input("\nNhan Enter de thoat...")
        return 1

    print(f"\n[+] Da tu dong phat hien bo mach ESP32 tai cong: {port}")

    # 3. Locate Binaries
    bootloader = REPO_ROOT / "release" / "firmware" / "bootloader.bin"
    partitions = REPO_ROOT / "release" / "firmware" / "partitions.bin"
    firmware = REPO_ROOT / "release" / "firmware" / "firmware.bin"

    for f in [bootloader, partitions, firmware]:
        if not f.exists():
            print(f"[!] Khong tim thay file binary: {f}")
            return 1

    # 4. Flash
    print(f"[+] Dang nap firmware vao {port} (Baudrate: 921600)... Vui long cho...")
    cmd = [
        "--chip", "esp32s3",
        "--port", str(port),
        "--baud", "921600",
        "write_flash", "-z",
        "0x0", str(bootloader),
        "0x8000", str(partitions),
        "0x10000", str(firmware)
    ]

    try:
        esptool.main(cmd)
        print("\n" + "=" * 60)
        print("  >>> [THANH CONG] Da nap firmware hoan tat vao ESP32-S3! <<<")
        print("  Hay rut cap hoac bam nut RST tren mach de khoi dong.")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n[!] Loi trong qua trinh nap: {e}")
        print("\n[MEO] Neu mach bi treo o 'Connecting...':")
        print("      Nhan giu nut BOOT tren mach -> Bam nut RST -> Tha nut BOOT -> Chay lai lenh.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
