from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import server  # noqa: E402
from pipelines import visual_qa as f1  # noqa: E402


def make_jpeg_bytes(color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def fake_wav_bytes() -> bytes:
    return b"RIFF$\x00\x00\x00WAVEfmt "


class Feature1GgufApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def patch_attr(self, obj, name: str, value) -> None:
        old_value = getattr(obj, name)
        setattr(obj, name, value)
        self.addCleanup(lambda: setattr(obj, name, old_value))

    def patch_success_pipeline(self, transcript: str = "Trước mặt tôi có gì?") -> list[str]:
        calls: list[str] = []

        def fake_stt(audio_path: Path, timings=None) -> str:
            calls.append("stt")
            if timings:
                timings.add("audio_preprocessing", 0.01)
                timings.add("stt", 0.02)
            return transcript

        def fake_translate(text: str, src_lang: str, timings=None, timing_key=None) -> str:
            calls.append(f"translate:{src_lang}")
            if timings:
                timings.add(timing_key or ("vi_to_en" if src_lang == "vi" else "en_to_vi"), 0.01)
            if src_lang == "vi":
                return "What is in front of me?"
            return "Phía trước có một cái bàn."

        def fake_vlm(image_path: Path, question_en: str, timings=None) -> str:
            calls.append("vlm")
            if timings:
                timings.add("image_preparation", 0.01)
                timings.add("vlm", 0.02)
            return "There is a table in front of you."

        def fake_tts(text: str, out_path: Path, timings=None) -> None:
            calls.append("tts")
            if timings:
                timings.add("tts", 0.01)
            out_path.write_bytes(fake_wav_bytes())

        self.patch_attr(server, "speech_to_text_vi", fake_stt)
        self.patch_attr(server, "translate_text", fake_translate)
        self.patch_attr(server, "run_vlm_question", fake_vlm)
        self.patch_attr(server, "synthesize_speech_vi", fake_tts)
        return calls

    def test_case1_text_question_keeps_mobile_contract(self) -> None:
        calls = self.patch_success_pipeline()

        response = self.client.post(
            "/qa",
            files={"image": ("scene.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"question": "Trước mặt tôi có gì?"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["answer"], "Phía trước có một cái bàn.")
        self.assertEqual(data["question"], "Trước mặt tôi có gì?")
        self.assertIn("timings", data)
        self.assertNotIn("audio_base64", data)
        self.assertEqual(calls, ["translate:vi", "vlm", "translate:en"])

    def test_case2_multiple_objects_question_uses_same_contract(self) -> None:
        self.patch_success_pipeline()

        response = self.client.post(
            "/qa",
            files={"image": ("objects.jpg", make_jpeg_bytes("blue"), "image/jpeg")},
            data={"question": "Có vật cản nào trước mặt tôi không?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question"], "Có vật cản nào trước mặt tôi không?")

    def test_case3_missing_or_invalid_image(self) -> None:
        missing = self.client.post("/qa", data={"question": "Trước mặt tôi có gì?"})
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.json()["detail"]["error"], "missing_image")

        invalid = self.client.post(
            "/qa",
            files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
            data={"question": "Trước mặt tôi có gì?"},
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json()["detail"]["error"], "invalid_image")

    def test_case4_audio_stt_empty_returns_422(self) -> None:
        self.patch_success_pipeline(transcript="")

        response = self.client.post(
            "/qa",
            files={
                "image": ("scene.jpg", make_jpeg_bytes(), "image/jpeg"),
                "audio": ("question.wav", fake_wav_bytes(), "audio/wav"),
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["error"], "empty_transcript")

    def test_case5_llama_server_unavailable_returns_503(self) -> None:
        self.patch_success_pipeline()

        def unavailable(image_path: Path, question_en: str, timings=None) -> str:
            if timings:
                timings.add("vlm", 0.01)
            raise f1.LlamaServerUnavailable("llama-server binary not found")

        self.patch_attr(server, "run_vlm_question", unavailable)

        response = self.client.post(
            "/qa",
            files={"image": ("scene.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"question": "Trước mặt tôi có gì?"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["error"], "llama_server_unavailable")

    def test_case6_llama_timeout_returns_504(self) -> None:
        self.patch_success_pipeline()

        def timeout(image_path: Path, question_en: str, timings=None) -> str:
            if timings:
                timings.add("vlm", 120.0)
            raise f1.LlamaServerTimeout("SmolVLM2 GGUF inference timed out")

        self.patch_attr(server, "run_vlm_question", timeout)

        response = self.client.post(
            "/qa",
            files={"image": ("scene.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"question": "Trước mặt tôi có gì?"},
        )

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["detail"]["error"], "qa_inference_timeout")

    def test_case7_audio_pipeline_returns_audio_and_vlm_cache_reuses_server(self) -> None:
        calls = self.patch_success_pipeline()

        response = self.client.post(
            "/qa",
            files={
                "image": ("scene.jpg", make_jpeg_bytes(), "image/jpeg"),
                "audio": ("question.wav", fake_wav_bytes(), "audio/wav"),
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("audio_base64", data)
        self.assertEqual(data["audio_mime"], "audio/wav")
        self.assertEqual(calls, ["stt", "translate:vi", "vlm", "translate:en", "tts"])

        ensure_calls = {"count": 0}
        info = {
            "url": "http://127.0.0.1:8085",
            "mode": "hf_repo",
            "model": "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
            "mmproj": None,
            "quantization": "auto",
            "server_bin": "llama-server",
            "auto_start": True,
            "context_size": 2048,
            "startup_timeout_s": 60,
            "request_timeout_s": 120,
            "openai_model": "SmolVLM2-256M",
            "healthy": True,
            "started_by_backend": False,
            "load_seconds": 0.0,
        }

        def fake_ensure() -> dict:
            ensure_calls["count"] += 1
            return info

        self.patch_attr(f1, "_vlm_server_info", None)
        self.patch_attr(f1, "ensure_llama_server", fake_ensure)
        self.patch_attr(f1, "_llama_server_is_up", lambda url: f1._vlm_server_info is not None)

        first = f1.get_vlm_model()
        second = f1.get_vlm_model()

        self.assertIs(first, second)
        self.assertEqual(ensure_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
