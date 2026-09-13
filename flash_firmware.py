import argparse
import configparser
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
FIRMWARE_DIR = REPO_ROOT / "firmware"
SOURCE_DIR = FIRMWARE_DIR / "esp32_sense"
BUILD_ENV = "esp32s3cam"
BUILD_DIR = FIRMWARE_DIR / ".pio" / "build" / BUILD_ENV
RELEASE_DIR = REPO_ROOT / "release" / "firmware"
MANIFEST_PATH = RELEASE_DIR / "manifest.json"
ARTIFACT_NAMES = ("bootloader.bin", "partitions.bin", "firmware.bin")
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".ino", ".s"}
ESP32_PORT_KEYWORDS = (
    "ch340",
    "ch343",
    "cp210",
    "ftdi",
    "usb-serial",
    "usb serial",
    "uart",
    "1a86",
    "10c4",
    "303a",
)


class FlashError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def firmware_source_files():
    files = [FIRMWARE_DIR / "platformio.ini"]
    files.extend(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    return sorted(files, key=lambda path: path.as_posix().lower())


def collect_source_state():
    files = firmware_source_files()
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FlashError(f"Firmware source is missing: {missing[0]}")

    state = {}
    for path in files:
        stat = path.stat()
        state[path.relative_to(REPO_ROOT).as_posix()] = {
            "sha256": sha256_file(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return state


def collect_artifact_state(directory):
    state = {}
    for name in ARTIFACT_NAMES:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FlashError(f"Artifact is missing or empty: {path}")
        state[name] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    return state


def artifact_status():
    missing = [name for name in ARTIFACT_NAMES if not (RELEASE_DIR / name).is_file()]
    if missing:
        return False, "missing release artifact(s): " + ", ".join(missing)
    if not MANIFEST_PATH.is_file():
        return False, "release manifest is missing"

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"release manifest is invalid: {exc}"

    if manifest.get("schema_version") != 1:
        return False, "release manifest schema is unsupported"
    if manifest.get("build_environment") != BUILD_ENV:
        return False, "release manifest build environment does not match"

    try:
        current_sources = collect_source_state()
        current_artifacts = collect_artifact_state(RELEASE_DIR)
    except FlashError as exc:
        return False, str(exc)

    if manifest.get("sources") != current_sources:
        return False, "firmware source content or metadata changed"
    if manifest.get("artifacts") != current_artifacts:
        return False, "release artifact hash or size changed"
    return True, "release artifacts match the current firmware source"


def platformio_command():
    executable = shutil.which("pio") or shutil.which("platformio")
    if not executable:
        raise FlashError("PlatformIO CLI was not found in PATH")
    return executable


def run_checked(command, description):
    print(f"[+] {description}")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise FlashError(f"{description} failed with exit code {completed.returncode}")


def build_release_artifacts():
    source_state_before = collect_source_state()
    run_checked(
        [platformio_command(), "run", "-d", str(FIRMWARE_DIR), "-e", BUILD_ENV],
        "Building firmware from the current source with PlatformIO",
    )
    source_state_after = collect_source_state()
    if source_state_after != source_state_before:
        raise FlashError("Firmware source changed during the build; refusing to publish or flash")

    collect_artifact_state(BUILD_DIR)
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_NAMES:
        source = BUILD_DIR / name
        destination = RELEASE_DIR / name
        temporary = RELEASE_DIR / f".{name}.tmp"
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    manifest = {
        "schema_version": 1,
        "build_environment": BUILD_ENV,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_state_after,
        "artifacts": collect_artifact_state(RELEASE_DIR),
    }
    temporary_manifest = RELEASE_DIR / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, MANIFEST_PATH)

    current, reason = artifact_status()
    if not current:
        raise FlashError(f"Published release artifacts failed validation: {reason}")
    print(f"[+] Published fresh artifacts to {RELEASE_DIR}")


def ensure_python_dependencies():
    missing = []
    try:
        import esptool  # noqa: F401
    except ImportError:
        missing.append("esptool")
    try:
        import serial  # noqa: F401
    except ImportError:
        missing.append("pyserial")

    if missing:
        print("[!] Installing missing Python dependencies: " + ", ".join(missing))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "esptool", "pyserial"]
        )


def configured_upload_port():
    config_path = FIRMWARE_DIR / "platformio.ini"
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    return parser.get(f"env:{BUILD_ENV}", "upload_port", fallback=None)


def port_label(port):
    return f"{port.device} ({port.description or 'unknown device'}; {port.hwid or 'no HWID'})"


def likely_esp32_port(port):
    identity = f"{port.description or ''} {port.hwid or ''}".lower()
    return any(keyword in identity for keyword in ESP32_PORT_KEYWORDS)


def select_esp32_port(explicit_port=None):
    import serial.tools.list_ports

    ports = list(serial.tools.list_ports.comports())
    by_device = {port.device.lower(): port for port in ports}

    if explicit_port:
        selected = by_device.get(explicit_port.lower())
        if not selected:
            available = ", ".join(port_label(port) for port in ports) or "none"
            raise FlashError(
                f"Requested port {explicit_port} is not present. Available ports: {available}"
            )
        return selected.device, "explicit --port"

    candidates = [port for port in ports if likely_esp32_port(port)]
    configured = configured_upload_port()
    if len(candidates) == 1:
        return candidates[0].device, "single detected ESP32 USB adapter"
    if configured:
        configured_device = by_device.get(configured.lower())
        if configured_device and configured_device in candidates:
            return configured_device.device, "PlatformIO upload_port matched an ESP32 adapter"
    if len(candidates) > 1:
        labels = "; ".join(port_label(port) for port in candidates)
        raise FlashError(
            "Multiple possible ESP32 serial devices were found: "
            f"{labels}. Select one with --port."
        )
    if configured:
        configured_device = by_device.get(configured.lower())
        if configured_device and "bluetooth" not in (configured_device.description or "").lower():
            return configured_device.device, "present PlatformIO upload_port"

    available = "; ".join(port_label(port) for port in ports) or "none"
    raise FlashError(f"No ESP32 USB serial adapter was identified. Available ports: {available}")


def flash_release_artifacts(port):
    collect_artifact_state(RELEASE_DIR)
    command = [
        sys.executable,
        "-m",
        "esptool",
        "--chip",
        "esp32s3",
        "--port",
        port,
        "--baud",
        "921600",
        "--after",
        "hard-reset",
        "write-flash",
        "-z",
        "0x0",
        str(RELEASE_DIR / "bootloader.bin"),
        "0x8000",
        str(RELEASE_DIR / "partitions.bin"),
        "0x10000",
        str(RELEASE_DIR / "firmware.bin"),
    ]
    run_checked(command, f"Flashing validated release artifacts to {port}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build when needed and flash BSmart ESP32-S3 firmware safely."
    )
    parser.add_argument("--port", help="Serial port to use, for example COM7")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Always rebuild and republish firmware before flashing",
    )
    return parser.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    print("=" * 64)
    print("       BSmart One-Click ESP32-S3 Firmware Flasher")
    print("=" * 64)

    try:
        ensure_python_dependencies()
        port, port_reason = select_esp32_port(args.port)
        print(f"[+] Selected serial port: {port} ({port_reason})")

        current, reason = artifact_status()
        if args.rebuild:
            print("[!] Rebuild requested with --rebuild")
            build_release_artifacts()
        elif not current:
            print(f"[!] Release artifacts are missing or stale: {reason}")
            build_release_artifacts()
        else:
            print(f"[+] {reason}; skipping PlatformIO build")

        current, reason = artifact_status()
        if not current:
            raise FlashError(f"Release artifacts are not safe to flash: {reason}")

        flash_release_artifacts(port)
        print("=" * 64)
        print("[SUCCESS] Firmware flashed and the ESP32-S3 was hard-reset.")
        print("=" * 64)
        return 0
    except (FlashError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
