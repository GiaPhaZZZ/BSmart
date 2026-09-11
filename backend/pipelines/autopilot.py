# ============================================================================
# Blind Navigation Vision Pipeline (v5.1)
#
# WHAT CHANGED FROM v4, AND WHY
# ----------------------------------------------------------------------------
# Added the stair/hole hazard classifier you trained
# (yolo26s_normal_stair_hole.pt, a YOLO26s *classification* checkpoint --
# same architecture family as the detector, different head) into the
# AUTOPILOT / continuous glasses loop ONLY (run_on_glasses, section 16).
# It is intentionally NOT wired into run_image_to_audio (the single-shot
# CLI path) -- you asked specifically for "Function 2 - autopilot".
#
# How it works, matching your notebook snippet 1:1:
#   - model.predict(source=frame, imgsz=224, device=..., verbose=False)
#   - results[0].probs.top1 / top1conf  (classification, not boxes)
#   - imgsz=224 is kept because that's what the checkpoint was trained/
#     evaluated at in your notebook -- changing it would shift the
#     accuracy numbers you already validated, so it's its own CONFIG key
#     (stair_hole_imgsz) instead of reusing zipdepth_input_size.
#
# Confidence gating ("say nothing" rule):
#   - StairHoleClassifier.classify() returns None (no hazard) unless
#     BOTH: (a) top1conf >= CONFIG["stair_hole_conf_threshold"], AND
#     (b) the winning class name actually contains "stair" or "hole".
#     A predicted "normal"/"none"/"flat"/etc. class, or a low-confidence
#     stair/hole prediction, produces no message at all -- it does not
#     fall back to any other text, per your "say nothing" requirement.
#   - When it *does* fire, it's turned into one extra spoken Vietnamese
#     message ("Có bậc thang." / "Có hố.") with its own cooldown, the
#     same pattern already used for the terrain
#     ("Địa hình chênh vênh...") message.
#
# UPDATE (v5.1): your hazard checkpoint was retrained with 3 classes --
# normal / stair / hole (previously it was a 2-class stair / pothole
# model). The checkpoint file is now yolo26s_normal_stair_hole.pt. Every
# "pothole" reference below (config keys, class names, variable names,
# the classifier class itself) has been renamed to "hole" to match the
# new class name -- e.g. StairPotholeClassifier -> StairHoleClassifier,
# CONFIG["stair_pothole_*"] -> CONFIG["stair_hole_*"]. The actual logic
# is unchanged: it still only speaks when the model is confident AND the
# winning class is recognizably "stair" or "hole" -- a predicted "normal"
# (or anything below threshold) still produces no message and is never
# added to the warning text.
#
# Everything else below (ground-plane depth calibration, per-object
# tracking, path advisor, TTS, etc.) is unchanged from v4.
# ============================================================================

import time
import wave
import numpy as np
import cv2
from dataclasses import dataclass, field
from collections import deque
from ultralytics import YOLO

# ZipDepth is a git checkout, not a pip package. Your working test_depth.py
# never hits this problem because it shells out to scripts/infer.py *from
# inside* the ZipDepth folder, which inserts its own parent onto sys.path
# (see scripts/infer.py: `sys.path.insert(0, str(Path(__file__).parent.parent))`).
# Since pipeline.py imports zipdepth directly, do the same trick here so it
# works whether or not `pip install -e ZipDepth` was ever run.
import sys
from pathlib import Path


def _ensure_zipdepth_importable():
    try:
        import zipdepth  # noqa: F401
        return
    except ImportError:
        pass
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "models" / "ZipDepth",
        Path(__file__).resolve().parent / "ZipDepth",
        Path.cwd() / "models" / "ZipDepth",
        Path.cwd() / "ZipDepth",
    ]
    for c in candidates:
        if (c / "zipdepth" / "__init__.py").exists():
            sys.path.insert(0, str(c))
            return
    raise ImportError(
        "Could not find the ZipDepth repo (looked for a 'zipdepth' package "
        f"under: {', '.join(str(c) for c in candidates)}). Either run "
        "`pip install -e ZipDepth` inside your glass venv, or make sure the "
        "ZipDepth/ folder sits under models/ or next to pipeline."
    )


_ensure_zipdepth_importable()
from zipdepth.inference.predictor import DepthInference

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _resolve_model(rel: str) -> str:
    p_models = MODELS_DIR / rel
    if p_models.exists():
        return str(p_models)
    p_root = REPO_ROOT / rel
    if p_root.exists():
        return str(p_root)
    return str(p_models)


# ============================================================================
# 1. CONFIG
# ============================================================================

CONFIG = {

    # ------------------------------------------------------------------
    # DETECTOR (YOLO26s -- object detection only, no depth head anymore)
    # ------------------------------------------------------------------
    "detector_conf": 0.35,
    "detector_model_path": _resolve_model("yolo26s.pt"),

    # ------------------------------------------------------------------
    # DEPTH (ZipDepth -- separate, real monocular depth model)
    # ------------------------------------------------------------------
    "zipdepth_checkpoint": _resolve_model("ZipDepth/checkpoints/zipdepth_base_npu.pth"),
    "zipdepth_variant": "base",           # must match the checkpoint
    "zipdepth_input_size": 384,           # ZipDepth's own default; raise for accuracy, lower for speed
    "zipdepth_ensure_multiple_of": 32,
    "zipdepth_device": "cpu",             # "cuda" if you have a GPU box for dev/testing
    "zipdepth_use_npu_upsample": True,    # must be True to match the _npu.pth checkpoint above

    # ------------------------------------------------------------------
    # CAMERA GEOMETRY
    # ------------------------------------------------------------------
    "horizontal_fov_deg": 68.0,

    # ---- ONE-TIME PHYSICAL CALIBRATION OF THE GLASSES, NOT PER-FRAME ----
    # These two numbers describe how the camera sits on the person's face;
    # they don't change frame to frame, only if the hardware mount changes.
    # How to set them:
    #   camera_height_m:  measure the lens height off the ground while the
    #                      glasses are worn normally, standing straight.
    #   camera_pitch_deg: put two markers on flat ground at known distances
    #                      straight ahead (e.g. 1m and 3m), take a shot, and
    #                      adjust this value until GroundPlaneModel's
    #                      expected-depth row for each marker's pixel row
    #                      matches the real measured distance. 0 = camera
    #                      looking dead level; positive = tilted down.
    "camera_height_m": 1.55,
    "camera_pitch_deg": 8.0,

    # ------------------------------------------------------------------
    # DISTANCE THRESHOLDS (now real meters, after ground-plane calibration)
    # ------------------------------------------------------------------
    "very_close_m": 1.0,
    "max_warning_m": 4.0,
    "overhead_max_m": 6.0,

    # "important"-tier classes (see CLASS_PRIORITY_VI) are only spoken
    # when closer than this or already DANGER -- keeps e.g. a distant
    # bench/dog from cluttering the audio while still warning if it's
    # actually in the way.
    "important_class_max_m": 2.0,

    # ------------------------------------------------------------------
    # MOTION / TTC
    # ------------------------------------------------------------------
    "ttc_danger_s": 1.5,
    "ttc_warning_s": 3.0,
    "approaching_speed_mps": 0.3,

    # ------------------------------------------------------------------
    # TRACKING
    # ------------------------------------------------------------------
    "max_center_dist_ratio": 0.08,
    "track_history_len": 8,
    "confirm_frames": 2,
    "max_missed_frames": 5,
    "max_track_match_time_s": 6.0,
    "emergency_distance_m": 1.2,

    "expected_speed_mps": {
        "car": 3.0, "bus": 3.0, "truck": 3.0, "motorcycle": 4.0,
        "bicycle": 3.5, "person": 1.6, "dog": 2.5, "horse": 2.5,
    },
    "default_expected_speed_mps": 1.5,

    # ------------------------------------------------------------------
    # LATENCY
    # ------------------------------------------------------------------
    "tts_onset_latency_s": 0.4,

    # ------------------------------------------------------------------
    # ANNOUNCEMENTS
    # ------------------------------------------------------------------
    "cooldown_s": {"DANGER": 0.0, "WARNING": 3.0},
    "crowd_min_count": 5,
    "max_announcements_per_cycle": 3,
    "min_gap_between_any_announcement_s": 0.4,

    # ------------------------------------------------------------------
    # FREE SPACE / WALKABLE PATH
    # ------------------------------------------------------------------
    "path_columns": 3,

    # Narrow band right in front of the person's next few steps -- used
    # for the final CLEAR/BLOCKED/STEP_DOWN verdict per column.
    "path_band_y_range": (0.55, 0.85),

    # Wide band used ONLY to *find* the ground plane each frame (more
    # samples = a more robust fit). Excludes anything above the
    # geometric horizon automatically, and excludes detected objects.
    "ground_calib_band_y_range": (0.35, 0.97),
    "ground_calib_min_inlier_frac": 0.5,
    "ground_calib_max_relative_jump": 0.6,   # reject a fit if |a| jumps >60% vs last good frame
    "ground_calib_max_stale_frames": 6,      # reuse the last good calibration for up to N frames

    "path_min_clear_m": 1.0,          # nearer than this in a column -> BLOCKED
    "path_drop_delta_m": 0.5,         # ground recedes this much past flat-plane model -> STEP_DOWN
    "path_bump_delta_m": 0.35,        # ground appears this much closer than flat-plane model -> BLOCKED
    "path_min_valid_px": 40,
    "path_cooldown_s": 4.0,

    # ------------------------------------------------------------------
    # STAIR / HOLE HAZARD CLASSIFIER (autopilot / run_on_glasses ONLY)
    # ------------------------------------------------------------------
    # Separate YOLO26s *classification* checkpoint (not the COCO detector
    # above). Whole-frame classify -> normal / stair / hole (3 classes,
    # matching your training notebook's stair_hole_normal_cls dataset).
    # Only used by run_on_glasses.
    "stair_hole_model_path": "yolo26s_normal_stair_hole.pt",
    "stair_hole_imgsz": 224,          # matches the notebook eval you validated it at
    "stair_hole_conf_threshold": 0.4,
    "stair_hole_cooldown_s": 4.0,
    # Any predicted class whose name is in this set (or that doesn't
    # contain "stair"/"hole" at all) is treated as "no hazard", no
    # matter how high its confidence is. "normal" is the actual trained
    # class name; the rest are kept as defensive fallbacks in case you
    # ever retrain with different "no hazard" folder names.
    "stair_hole_ignore_labels": {"normal", "none", "flat", "background", "other", "ground", "safe"},

    # ------------------------------------------------------------------
    # PATH VISUALIZATION
    # ------------------------------------------------------------------
    "draw_path": True,
    "path_overlay_alpha": 0.30,
    "path_visual_y_start": 0.55,

    # ------------------------------------------------------------------
    # BATTERY-AWARE CAPTURE
    # ------------------------------------------------------------------
    "capture_interval_s": {"calm": 3.0, "alert": 1.0},
    "alert_hold_cycles": 4,

    # ------------------------------------------------------------------
    # SPEECH
    # ------------------------------------------------------------------
    "max_speech_queue_s": 6.0,
    "tts_chars_per_second": 14.0,
    "piper_voice_path": _resolve_model("voices/vi_VN-vais1000-medium.onnx"),  # from your download_models.py step
}


# ============================================================================
# 2. COCO CLASSES / VIETNAMESE LABELS  (unchanged from v3/v4)
# ============================================================================

KEEP_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    15: "cat", 16: "dog", 17: "horse", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 56: "chair", 57: "couch",
    59: "bed", 60: "dining table", 61: "toilet", 71: "sink", 72: "refrigerator",
    24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase", 39: "bottle",
    63: "laptop", 67: "cell phone",
}

DYNAMIC_HAZARD_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle", "dog", "horse", "person"}

CLASS_VI = {
    "person": "người", "bicycle": "xe đạp", "car": "ô tô", "motorcycle": "xe máy",
    "bus": "xe buýt", "truck": "xe tải", "cat": "mèo", "dog": "chó", "horse": "ngựa",
    "traffic light": "đèn giao thông", "fire hydrant": "trụ cứu hỏa", "stop sign": "biển dừng",
    "parking meter": "đồng hồ đỗ xe", "bench": "ghế dài", "chair": "ghế", "couch": "ghế sofa",
    "bed": "giường", "dining table": "bàn ăn", "toilet": "bồn cầu", "sink": "bồn rửa",
    "refrigerator": "tủ lạnh", "backpack": "ba lô", "umbrella": "ô dù", "handbag": "túi xách",
    "suitcase": "vali", "bottle": "chai", "laptop": "laptop", "cell phone": "điện thoại",
}

BROAD_SIDE = {"FAR LEFT": "LEFT", "LEFT": "LEFT", "AHEAD": "AHEAD", "RIGHT": "RIGHT", "FAR RIGHT": "RIGHT"}
BROAD_SIDE_VI = {"LEFT": "bên trái", "AHEAD": "phía trước", "RIGHT": "bên phải"}
HEIGHT_VI = {"HEAD-LEVEL": "trên đầu", "GROUND-LEVEL": "trên mặt đất", "BODY-LEVEL": "ngang ngực"}
WARNING_VI = {"DANGER": "NGUY HIỂM", "WARNING": "CHÚ Ý"}

# Vietnamese phrasing for the stair/hole hazard classifier. Keyed by the
# normalized hazard_key ("stair" / "hole") that StairHoleClassifier
# resolves the model's raw class name to -- see that class below.
# Kept short and direct on purpose (no "Phía trước" / "ahead" prefix) --
# just "có bậc thang" / "có hố". _build_stair_hole_message capitalizes
# the first letter since it's spoken as its own sentence.
STAIR_HOLE_PHRASE_VI = {
    "stair": "có bậc thang",
    "hole": "có hố",
}

# ------------------------------------------------------------------
# ANNOUNCEMENT PRIORITY BY CLASS
#   critical  -> always announced (moving hazards + people)
#   important -> only announced when close (blocking) or already DANGER
#   low / unlisted -> never announced in the spoken audio (still shows
#                      up in the raw DETECTIONS debug printout)
# ------------------------------------------------------------------
CLASS_PRIORITY_VI = {
    "person": "critical", "car": "critical", "motorcycle": "critical",
    "bicycle": "critical", "bus": "critical", "truck": "critical",
    "dog": "important", "cat": "important", "horse": "important",
    "bench": "important", "chair": "important", "couch": "important",
    "bed": "important", "dining table": "important",
    "traffic light": "low", "fire hydrant": "low", "stop sign": "low",
    "parking meter": "low", "toilet": "low", "sink": "low",
    "refrigerator": "low", "backpack": "low", "umbrella": "low",
    "handbag": "low", "suitcase": "low", "bottle": "low",
    "laptop": "low", "cell phone": "low",
}


def _capitalize_vi(text):
    return text[0].upper() + text[1:] if text else text


APPROACHING_VI = "đang tiến lại gần"
CROWD_VI = "NGUY HIỂM đám đông phía trước"


def _path_side_name(index, n_cols):
    """Maps a path column index to a spoken side name. Works for any
    CONFIG['path_columns'], not just 3 -- first column is always 'bên
    trái', last is always 'bên phải', anything in between is 'trước mặt'."""
    if n_cols == 1:
        return "trước mặt"
    if index == 0:
        return "bên trái"
    if index == n_cols - 1:
        return "bên phải"
    return "trước mặt"


def build_path_phrase_vi(path_status, include_clear=True):
    """Turns the per-column CLEAR/BLOCKED/STEP_DOWN/UNKNOWN verdicts into one
    natural Vietnamese sentence, naming every affected side instead of only
    looking at the middle column:
        path_status is None    -> ground-plane calibration failed this frame
                                   (e.g. noisy night scene) -- we genuinely
                                   don't know, and say so rather than
                                   staying silent (silence would read as
                                   "path is fine" to a blind user).
        all clear               -> "Đường bằng phẳng" (or None if include_clear=False)
        one/more sides bad      -> "Địa hình chênh vênh bên trái và trước mặt"
    UNKNOWN columns are treated like CLEAR (not enough signal to warn)."""
    if path_status is None:
        return "Chưa xác định được địa hình phía trước, cẩn trọng bước đi." if include_clear else None
    if not path_status:
        return "Đường bằng phẳng" if include_clear else None

    n = len(path_status)
    bad_sides = []
    for i, (status, _) in enumerate(path_status):
        if status in ("BLOCKED", "STEP_DOWN"):
            side = _path_side_name(i, n)
            if side not in bad_sides:
                bad_sides.append(side)

    if not bad_sides:
        return "Đường bằng phẳng" if include_clear else None

    if len(bad_sides) == 1:
        joined = bad_sides[0]
    else:
        joined = ", ".join(bad_sides[:-1]) + " và " + bad_sides[-1]
    return f"Địa hình chênh vênh {joined}"


# ============================================================================
# 3. GEOMETRY HELPERS
# ============================================================================

def focal_length_px(image_width, hfov_deg):
    return (image_width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)


def get_horizontal_position(center_x, image_width):
    x_ratio = center_x / image_width
    angle_offset = (x_ratio - 0.5) * CONFIG["horizontal_fov_deg"]
    if x_ratio < 0.20:
        zone = "FAR LEFT"
    elif x_ratio < 0.40:
        zone = "LEFT"
    elif x_ratio > 0.80:
        zone = "FAR RIGHT"
    elif x_ratio > 0.60:
        zone = "RIGHT"
    else:
        zone = "AHEAD"
    return zone, angle_offset


def get_height_zone(y1, y2, image_height):
    # NOTE: this used to depend on `camera_mount_height_ratio` (a made-up
    # fraction of image height). It's now driven by the real ground-plane
    # geometry instead -- see GroundPlaneModel.horizon_row().
    horizon_row = GroundPlaneModel.horizon_row_static(image_height)
    if (y2 / image_height) > 0.85:
        return "GROUND-LEVEL"
    if y2 < horizon_row:
        return "HEAD-LEVEL"
    return "BODY-LEVEL"


def geometric_distance_proxy(x1, y1, x2, y2, image_w, image_h):
    """Crude last-resort distance guess, used only when depth calibration
    has never succeeded (e.g. very first frame, all-sky scene)."""
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(image_w * image_h)
    size_cue = min(1.0, area_ratio * 30.0)
    bottom_position = y2 / image_h
    closeness = 0.6 * size_cue + 0.4 * bottom_position
    return float(np.clip(closeness, 0.0, 1.0))


def format_distance_vi(distance_m):
    if distance_m >= 1.0:
        return f"{max(round(distance_m), 1)} mét"
    return f"{max(round((distance_m * 100) / 10) * 10, 10)} cm"


# ============================================================================
# 4. GROUND-PLANE MODEL  (the actual fix for the "always blocked" bug)
# ============================================================================

class GroundPlaneModel:
    """
    Computes, from camera geometry alone, the metric depth a perfectly flat
    ground plane would produce at every image row -- then robustly fits
    ZipDepth's raw (affine-invariant, unitless) output to that curve each
    frame to recover a real per-frame (scale, shift). This is the classic
    v-disparity / inverse-perspective-mapping idea used for curb and
    pothole detection.
    """

    def __init__(self, hfov_deg, camera_height_m, camera_pitch_deg):
        self.hfov_deg = hfov_deg
        self.camera_height_m = camera_height_m
        self.camera_pitch_deg = camera_pitch_deg
        self._expected_rows_cache = {}   # (h, w) -> expected metric depth per row
        self.last_good_ab = None         # (a, b)
        self.frames_since_good = 999

    @staticmethod
    def horizon_row_static(image_height, pitch_deg=CONFIG["camera_pitch_deg"]):
        """Rough pixel row of the horizon, used only for the HEAD-LEVEL /
        BODY-LEVEL height split (an approximation is fine there)."""
        cy = image_height / 2.0
        # A downward pitch pushes the horizon UP in the image (smaller row).
        return cy  # pitch shifts this slightly; kept simple since it's only
                   # used as a coarse head/body split, not a metric quantity.

    def expected_depth_rows(self, image_h, image_w):
        key = (image_h, image_w)
        cached = self._expected_rows_cache.get(key)
        if cached is not None:
            return cached

        fy = focal_length_px(image_w, self.hfov_deg)  # square-pixel assumption
        cy = image_h / 2.0
        rows = np.arange(image_h)
        angle_from_center = np.arctan((rows - cy) / fy)
        angle_below_horizon = np.radians(self.camera_pitch_deg) + angle_from_center

        depth = np.full(image_h, np.inf, dtype=np.float64)
        valid = angle_below_horizon > np.radians(0.5)   # ray must actually point at the ground
        depth[valid] = self.camera_height_m / np.tan(angle_below_horizon[valid])

        self._expected_rows_cache[key] = depth
        return depth

    @staticmethod
    def _robust_affine_fit(raw_pred, expected_inv_depth_map, sample_mask,
                            min_inlier_frac, max_relative_jump, prev_ab):
        """Fit raw_pred ~= a * expected_inv_depth + b via a Theil-Sen seed
        + MAD-based iteratively re-weighted least squares. Returns
        (a, b, inlier_count) or None if the ground band doesn't look
        trustworthy this frame (e.g. dominated by obstacles)."""
        m = sample_mask & np.isfinite(expected_inv_depth_map) & (expected_inv_depth_map > 0)
        x0 = expected_inv_depth_map[m].astype(np.float64)
        y0 = raw_pred[m].astype(np.float64)
        n_total = len(x0)
        if n_total < 200:
            return None

        rng = np.random.default_rng(0)
        n_pairs = min(4000, n_total * 2)
        i1 = rng.integers(0, n_total, n_pairs)
        i2 = rng.integers(0, n_total, n_pairs)
        dx = x0[i1] - x0[i2]
        ok = np.abs(dx) > 1e-6
        if ok.sum() < 20:
            return None
        slopes = (y0[i1[ok]] - y0[i2[ok]]) / dx[ok]
        a = float(np.median(slopes))
        b = float(np.median(y0 - a * x0))

        x, y = x0, y0
        for sigma_k in (3.0, 2.5, 2.0, 1.5, 1.5, 1.5):
            resid = y - (a * x + b)
            sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
            keep = np.abs(resid - np.median(resid)) < sigma_k * sigma
            if keep.sum() < max(200, 0.05 * n_total):
                break
            x, y = x[keep], y[keep]
            A = np.stack([x, np.ones_like(x)], axis=1)
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            a, b = float(sol[0]), float(sol[1])

        inlier_frac = len(x) / n_total
        if inlier_frac < min_inlier_frac or a <= 0:
            return None

        if prev_ab is not None:
            prev_a, _ = prev_ab
            if prev_a > 0 and not (prev_a * (1 - max_relative_jump) <= a <= prev_a * (1 + max_relative_jump)):
                return None

        return a, b, int(len(x))

    def calibrate(self, raw_pred, detections, image_w, image_h):
        """Returns (a, b, is_stale) using this frame's data, falling back to
        the last good calibration for a few frames if this one is untrustworthy,
        so a single bad frame doesn't blind the announcer entirely."""
        expected_rows = self.expected_depth_rows(image_h, image_w)
        expected_inv_rows = np.zeros(image_h)
        finite = np.isfinite(expected_rows) & (expected_rows > 0)
        expected_inv_rows[finite] = 1.0 / expected_rows[finite]
        expected_inv_map = np.broadcast_to(expected_inv_rows[:, None], (image_h, image_w))

        y_lo, y_hi = CONFIG["ground_calib_band_y_range"]
        band_y1, band_y2 = int(image_h * y_lo), int(image_h * y_hi)
        sample_mask = np.zeros((image_h, image_w), dtype=bool)
        sample_mask[band_y1:band_y2, :] = True
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            sample_mask[y1:y2, x1:x2] = False  # objects aren't ground

        fit = self._robust_affine_fit(
            raw_pred, expected_inv_map, sample_mask,
            CONFIG["ground_calib_min_inlier_frac"],
            CONFIG["ground_calib_max_relative_jump"],
            self.last_good_ab,
        )

        if fit is not None:
            a, b, _ = fit
            self.last_good_ab = (a, b)
            self.frames_since_good = 0
            return a, b, False

        self.frames_since_good += 1
        if self.last_good_ab is not None and self.frames_since_good <= CONFIG["ground_calib_max_stale_frames"]:
            a, b = self.last_good_ab
            return a, b, True

        return None  # no usable calibration at all (e.g. very first frame, all sky)

    @staticmethod
    def to_metric(raw_pred, a, b):
        metric_inv = (raw_pred - b) / a
        return 1.0 / np.clip(metric_inv, 1e-3, None)


# ============================================================================
# 5. DEPTH ESTIMATOR (ZipDepth wrapper)
# ============================================================================

class ZipDepthEstimator:
    def __init__(self, checkpoint_path, variant, device, input_size,
                 ensure_multiple_of, use_npu_upsample):
        self.predictor = DepthInference(
            checkpoint_path=checkpoint_path,
            variant=variant,
            device=device,
            input_size=input_size,
            ensure_multiple_of=ensure_multiple_of,
            upsample_unfold=not use_npu_upsample,
            warmup_iters=0,
        )

    def estimate_raw(self, image_bgr):
        """Returns ZipDepth's RAW affine-invariant inverse-depth map,
        [H, W] float32, at the original image resolution. NOT metric --
        must go through GroundPlaneModel.calibrate()/to_metric() first."""
        return self.predictor.infer_image(image_bgr)


# ============================================================================
# 6. YOLO26s DETECTOR (detection only)
# ============================================================================

class Yolo26Detector:
    def __init__(self, model_path, class_map):
        self.model = YOLO(model_path)
        self.class_map = class_map

    def detect(self, image, conf):
        results = self.model(image, conf=conf, verbose=False)[0]
        raw = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in self.class_map:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            raw.append({
                "class": self.class_map[class_id],
                "confidence": float(box.conf[0]),
                "bbox_raw": (float(x1), float(y1), float(x2), float(y2)),
            })
        return raw


# ============================================================================
# 6b. STAIR / HOLE HAZARD CLASSIFIER  (autopilot only -- new in v5)
# ============================================================================

class StairHoleClassifier:
    """
    Wraps the separate YOLO26s *classification* checkpoint
    (yolo26s_normal_stair_hole.pt) you trained on normal / stair / hole
    crops -- this is NOT the COCO object detector (Yolo26Detector, section
    6). It just answers "does this whole frame look like a stair or a
    hole hazard", the same way your notebook uses it:

        results = model.predict(source=image, imgsz=224, device=..., verbose=False)
        probs = results[0].probs
        label, confidence = results[0].names[probs.top1], float(probs.top1conf)

    Used ONLY by the autopilot / continuous glasses loop (run_on_glasses,
    section 16) -- see run_image_to_audio's docstring/comment for why it's
    kept out of the single-image path.

    "Say nothing" behavior lives entirely in classify(): it returns None
    (no hazard) whenever confidence is below threshold OR the winning
    class isn't recognizably "stair" or "hole" -- there is no other
    fallback text produced from this classifier. In particular, a
    predicted "normal" class (whatever its confidence) always returns
    None and is never added to the warning text.
    """

    def __init__(self, model_path, imgsz, conf_threshold, ignore_labels, device="cpu", verbose_print=True):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.ignore_labels = {s.lower() for s in ignore_labels}
        self.device = device
        # New: print the raw classify() result every cycle (label + confidence
        # + pass/fail) purely so you can eyeball the model's live behavior --
        # this is independent of whether it ends up in the spoken warning.
        self.verbose_print = verbose_print

    def classify(self, image_bgr):
        """Returns (hazard_key, raw_label, confidence) if this frame is a
        confident stair/hole hazard, else None.

        hazard_key is normalized to 'stair' or 'hole' via a substring
        match on the model's own class name, so this keeps working
        whichever exact folder names you trained on (e.g. 'stairs',
        'staircase', 'hole', 'holes', ...). Anything else (a 'normal'/
        'flat'/'background'-type class, or low confidence) -> None.

        Regardless of the outcome, every call is printed (if verbose_print)
        so you can evaluate the raw model output frame-by-frame -- the
        warning/announcement logic downstream is unaffected by this and
        still only fires when the threshold is actually reached.
        """
        results = self.model.predict(
            source=image_bgr, imgsz=self.imgsz, device=self.device, verbose=False,
        )
        if not results or results[0].probs is None:
            if self.verbose_print:
                print("[stair/hole] no prediction returned this frame")
            return None

        probs = results[0].probs
        raw_label = results[0].names[int(probs.top1)]
        confidence = float(probs.top1conf)
        label_lower = raw_label.lower()

        below_threshold = confidence < self.conf_threshold
        ignored_class = label_lower in self.ignore_labels
        not_hazard_word = ("stair" not in label_lower) and ("hole" not in label_lower)

        if below_threshold:
            verdict = f"below threshold ({self.conf_threshold:.2f}) -- no warning"
        elif ignored_class:
            verdict = "ignored class -- no warning"
        elif not_hazard_word:
            verdict = "not a stair/hole class -- no warning"
        else:
            verdict = "HAZARD -- added to warning"

        if self.verbose_print:
            print(f"[stair/hole] label={raw_label!r} conf={confidence:.3f} -> {verdict}")

        if below_threshold or ignored_class or not_hazard_word:
            return None

        hazard_key = "stair" if "stair" in label_lower else "hole"
        return hazard_key, raw_label, confidence


# ============================================================================
# 7. OBJECT DISTANCE FROM (CALIBRATED, METRIC) DEPTH MAP
# ============================================================================

def robust_object_distance(metric_depth_map, x1, y1, x2, y2, height_zone, image_w, image_h):
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 2 or box_h <= 2:
        return None

    crop_x1, crop_x2 = int(x1 + box_w * 0.25), int(x2 - box_w * 0.25)
    crop_y1, crop_y2 = int(y1 + box_h * 0.25), int(y2 - box_h * 0.25)
    crop = metric_depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
    valid_crop = crop[np.isfinite(crop) & (crop > 0)]
    if len(valid_crop) == 0:
        return None
    median_depth = float(np.median(valid_crop))

    if height_zone == "HEAD-LEVEL":
        return median_depth

    cx = int((x1 + x2) / 2)
    patch_half_w = max(2, int(box_w * 0.1))
    py1, py2 = max(0, y2 - 4), min(image_h, y2 + 4)
    px1, px2 = max(0, cx - patch_half_w), min(image_w, cx + patch_half_w)
    patch = metric_depth_map[py1:py2, px1:px2]
    valid_patch = patch[np.isfinite(patch) & (patch > 0)]
    if len(valid_patch) == 0:
        return median_depth
    contact_depth = float(np.median(valid_patch))
    return 0.4 * median_depth + 0.6 * contact_depth


# ============================================================================
# 8. TRACK / TRACKER / TTC / WARNING LEVEL / PRIORITY  (unchanged from v3/v4)
# ============================================================================

@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: tuple
    position: str
    broad_side: str
    height_zone: str
    last_seen_ts: float = 0.0
    distance_history: deque = field(default_factory=lambda: deque(maxlen=CONFIG["track_history_len"]))
    timestamp_history: deque = field(default_factory=lambda: deque(maxlen=CONFIG["track_history_len"]))
    seen_streak: int = 1
    missed_streak: int = 0
    confirmed: bool = False
    last_announced_ts: float = -1e9
    last_announced_level: str = ""


class ObjectTracker:
    def __init__(self):
        self.tracks = {}
        self._next_id = 1

    @staticmethod
    def _iou(b1, b2):
        ax1, ay1, ax2, ay2 = b1
        bx1, by1, bx2, by2 = b2
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _match_gate(track, dt, diag, focal_px):
        base = CONFIG["max_center_dist_ratio"] * diag
        speed = CONFIG["expected_speed_mps"].get(track.class_name, CONFIG["default_expected_speed_mps"])
        distance = track.distance_history[-1] if track.distance_history else 3.0
        angular_disp = (speed * dt) / max(distance, 0.3)
        predicted = focal_px * angular_disp * 1.5
        return min(max(base, predicted), 0.6 * diag)

    def update(self, detections, image_w, image_h, timestamp):
        diag = float(np.hypot(image_w, image_h))
        focal_px = focal_length_px(image_w, CONFIG["horizontal_fov_deg"])
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        for track_id, track in list(self.tracks.items()):
            dt = timestamp - track.last_seen_ts
            if dt > CONFIG["max_track_match_time_s"]:
                continue
            max_center_dist = self._match_gate(track, dt, diag, focal_px)
            best_idx, best_score = None, -1.0
            tx1, ty1, tx2, ty2 = track.bbox
            t_cx, t_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2

            for i in unmatched_dets:
                det = detections[i]
                if det["class"] != track.class_name:
                    continue
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                center_dist = float(np.hypot(cx - t_cx, cy - t_cy))
                if center_dist > max_center_dist:
                    continue
                iou = self._iou((tx1, ty1, tx2, ty2), (x1, y1, x2, y2))
                score = iou - center_dist / diag
                if score > best_score:
                    best_score, best_idx = score, i

            if best_idx is not None:
                det = detections[best_idx]
                track.bbox = det["bbox"]
                track.position = det["position"]
                track.broad_side = det["broad_side"]
                track.height_zone = det["height_zone"]
                track.last_seen_ts = timestamp
                if det["distance_m"] is not None:
                    track.distance_history.append(det["distance_m"])
                    track.timestamp_history.append(timestamp)
                track.seen_streak += 1
                track.missed_streak = 0
                if track.seen_streak >= CONFIG["confirm_frames"]:
                    track.confirmed = True
                det["track_id"] = track_id
                unmatched_dets.remove(best_idx)
                matched_track_ids.add(track_id)

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_track_ids:
                track.missed_streak += 1
                track.seen_streak = 0
                if track.missed_streak > CONFIG["max_missed_frames"]:
                    del self.tracks[track_id]

        for i in unmatched_dets:
            det = detections[i]
            track_id = self._next_id
            self._next_id += 1
            track = Track(
                track_id=track_id, class_name=det["class"], bbox=det["bbox"],
                position=det["position"], broad_side=det["broad_side"],
                height_zone=det["height_zone"], last_seen_ts=timestamp,
            )
            if det["distance_m"] is not None:
                track.distance_history.append(det["distance_m"])
                track.timestamp_history.append(timestamp)
            track.confirmed = track.seen_streak >= CONFIG["confirm_frames"]
            self.tracks[track_id] = track
            det["track_id"] = track_id

        return self.tracks


def compute_ttc(track):
    if len(track.distance_history) < 2:
        return 0.0, None
    d0, d1 = track.distance_history[0], track.distance_history[-1]
    t0, t1 = track.timestamp_history[0], track.timestamp_history[-1]
    dt = t1 - t0
    if dt <= 0:
        return 0.0, None
    closing_speed = (d0 - d1) / dt
    if closing_speed <= CONFIG["approaching_speed_mps"]:
        return closing_speed, None
    ttc = track.distance_history[-1] / closing_speed
    return closing_speed, ttc


def get_warning_level(distance, height_zone, ttc):
    max_dist = CONFIG["overhead_max_m"] if height_zone == "HEAD-LEVEL" else CONFIG["max_warning_m"]
    ttc_danger = ttc is not None and ttc < CONFIG["ttc_danger_s"]
    ttc_warning = ttc is not None and ttc < CONFIG["ttc_warning_s"]
    if distance < CONFIG["very_close_m"] or ttc_danger:
        return "DANGER"
    if distance <= max_dist or ttc_warning:
        return "WARNING"
    return "SAFE"


def priority_score(track, warning_level, ttc):
    score = track.distance_history[-1] if track.distance_history else 999.0
    if track.height_zone == "HEAD-LEVEL":
        score -= 2.0
    if warning_level == "DANGER":
        score -= 5.0
    if track.class_name in DYNAMIC_HAZARD_CLASSES:
        score -= 0.5
    if ttc is not None:
        score -= max(0.0, CONFIG["ttc_warning_s"] - ttc)
    return score


# ============================================================================
# 9. FREE SPACE PATH ADVISOR (residual-based, ground-plane aware)
# ============================================================================

class FreeSpacePathAdvisor:
    def analyze(self, metric_depth_map, expected_depth_rows, image_w, image_h):
        y_lo, y_hi = CONFIG["path_band_y_range"]
        band_y1, band_y2 = int(image_h * y_lo), int(image_h * y_hi)
        n_cols = CONFIG["path_columns"]
        edges = np.linspace(0, image_w, n_cols + 1).astype(int)

        exp_band = expected_depth_rows[band_y1:band_y2][:, None]
        status = []

        for i in range(n_cols):
            x1, x2 = edges[i], edges[i + 1]
            band = metric_depth_map[band_y1:band_y2, x1:x2]
            exp = np.broadcast_to(exp_band, band.shape)
            valid = np.isfinite(band) & (band > 0) & np.isfinite(exp)

            if valid.sum() < CONFIG["path_min_valid_px"]:
                status.append(("UNKNOWN", None))
                continue

            actual = band[valid]
            expected = exp[valid]
            residual = float(np.median(actual - expected))
            near_depth = float(np.percentile(actual, 15))

            if near_depth < CONFIG["path_min_clear_m"] or residual < -CONFIG["path_bump_delta_m"]:
                status.append(("BLOCKED", near_depth))
            elif residual > CONFIG["path_drop_delta_m"]:
                status.append(("STEP_DOWN", near_depth))
            else:
                status.append(("CLEAR", near_depth))

        return status


# ============================================================================
# 10. PATH VISUALIZATION
# ============================================================================

def draw_walkable_path(image, path_status):
    if path_status is None or not CONFIG["draw_path"]:
        return image

    height, width = image.shape[:2]
    n_cols = len(path_status)
    y_start = int(height * CONFIG["path_visual_y_start"])
    y_bottom = height
    column_width = width / n_cols
    overlay = image.copy()

    path_colors = {"CLEAR": (0, 200, 0), "BLOCKED": (0, 0, 255),
                   "STEP_DOWN": (0, 140, 255), "UNKNOWN": (0, 220, 255)}
    path_labels = {"CLEAR": "WALKABLE", "BLOCKED": "BLOCKED",
                   "STEP_DOWN": "DROP / STEP", "UNKNOWN": "UNKNOWN"}

    for i, (status, near_depth) in enumerate(path_status):
        x1, x2 = int(i * column_width), int((i + 1) * column_width)
        color = path_colors.get(status, (255, 255, 255))
        cv2.rectangle(overlay, (x1, y_start), (x2, y_bottom), color, -1)
        cv2.rectangle(image, (x1, y_start), (x2, y_bottom), color, 3)

        label = path_labels.get(status, status)
        if near_depth is not None:
            label += f" {near_depth:.1f}m"
        cv2.putText(image, label, (x1 + 10, y_start + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    image = cv2.addWeighted(overlay, CONFIG["path_overlay_alpha"], image,
                             1.0 - CONFIG["path_overlay_alpha"], 0)

    clear_indices = [i for i, (status, _) in enumerate(path_status) if status == "CLEAR"]
    recommended = None
    if clear_indices:
        center_index = (n_cols - 1) / 2.0
        recommended = min(clear_indices, key=lambda i: abs(i - center_index))

    if recommended is not None:
        center_x = int((recommended + 0.5) * column_width)
        cv2.arrowedLine(image, (center_x, height - 30), (center_x, y_start + 55),
                         (0, 255, 0), 6, tipLength=0.08)
        if recommended == 0:
            direction_text = "WALK LEFT"
        elif recommended == n_cols - 1:
            direction_text = "WALK RIGHT"
        else:
            direction_text = "WALK AHEAD"
        cv2.rectangle(image, (10, 10), (300, 55), (0, 0, 0), -1)
        cv2.putText(image, direction_text, (20, 43), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(image, (10, 10), (390, 55), (0, 0, 0), -1)
        cv2.putText(image, "NO CLEAR PATH", (20, 43), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 0, 255), 2, cv2.LINE_AA)

    return image


ZONE_COLORS = {"HEAD-LEVEL": (255, 0, 255), "BODY-LEVEL": (0, 255, 255), "GROUND-LEVEL": (0, 165, 255)}


def draw_detections(image, tracks):
    for track in tracks.values():
        if not track.confirmed or not track.distance_history:
            continue
        x1, y1, x2, y2 = track.bbox
        distance = track.distance_history[-1]
        _, ttc = compute_ttc(track)
        warning_level = get_warning_level(distance, track.height_zone, ttc)
        color = (0, 0, 255) if warning_level == "DANGER" else ZONE_COLORS[track.height_zone]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        ttc_txt = f" ttc={ttc:.1f}s" if ttc is not None else ""
        label = f"#{track.track_id} {track.class_name} {distance:.2f}m {track.position}{ttc_txt}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y1 - 10, th + 10)
        cv2.rectangle(image, (x1, label_y - th - baseline), (x1 + tw, label_y + baseline), (0, 0, 0), -1)
        cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return image


def draw_navigation_output(image, tracks, path_status):
    image = draw_walkable_path(image, path_status)
    image = draw_detections(image, tracks)
    return image


# ============================================================================
# 11. ANNOUNCEMENT MANAGER
# ============================================================================

class AnnouncementManager:
    def __init__(self):
        self.last_any_announcement_ts = -1e9
        self.path_cooldowns = {}   # shared cooldown store: "path_blocked", "stair_hole:stair", etc.

    def build_messages(self, tracks, now, extra_latency_s=0.0, path_status=None,
                        stair_hole_hazard=None):
        candidates = []
        person_confirmed = 0

        for track in tracks.values():
            if not track.distance_history:
                continue
            distance = track.distance_history[-1]
            is_emergency = distance < CONFIG["emergency_distance_m"]
            if not track.confirmed and not is_emergency:
                continue
            closing_speed, ttc_raw = compute_ttc(track)
            ttc = None if ttc_raw is None else (ttc_raw - extra_latency_s)
            warning_level = get_warning_level(distance, track.height_zone, ttc)
            if warning_level == "SAFE":
                continue
            if track.class_name == "person":
                person_confirmed += 1
            # Class-priority gate: "low"-tier objects (bags, bottles, static
            # street furniture, ...) never make it into the spoken output;
            # "important"-tier objects (dog, chair, ...) only do when close
            # enough to actually be in the way, or already DANGER.
            tier = CLASS_PRIORITY_VI.get(track.class_name, "low")
            if tier == "low":
                continue
            if tier == "important" and warning_level != "DANGER" and distance > CONFIG["important_class_max_m"]:
                continue
            candidates.append((track, warning_level, distance, closing_speed, ttc))

        messages = []
        crowd_triggered = person_confirmed >= CONFIG["crowd_min_count"]
        if crowd_triggered:
            messages.append({"text": CROWD_VI, "priority": -100.0, "key": "CROWD", "warning": "DANGER"})

        scored = []
        for track, warning_level, distance, closing_speed, ttc in candidates:
            if crowd_triggered and track.class_name == "person":
                continue
            cooldown = CONFIG["cooldown_s"].get(warning_level, 3.0)
            if now - track.last_announced_ts < cooldown:
                continue
            score = priority_score(track, warning_level, ttc)
            scored.append((score, track, warning_level, distance, closing_speed, ttc))

        scored.sort(key=lambda t: t[0])

        for score, track, warning_level, distance, closing_speed, ttc in scored:
            if len(messages) >= CONFIG["max_announcements_per_cycle"]:
                break
            if now - self.last_any_announcement_ts < CONFIG["min_gap_between_any_announcement_s"]:
                break
            clause = self._format_clause(track, distance, closing_speed)
            text = _capitalize_vi(clause) + "."
            messages.append({
                "text": text, "clause": clause, "priority": score,
                "key": f"track:{track.track_id}", "warning": warning_level,
                "class_name": track.class_name, "broad_side": track.broad_side,
                "distance": distance,
            })
            track.last_announced_ts = now
            track.last_announced_level = warning_level
            self.last_any_announcement_ts = now

        if path_status and len(messages) < CONFIG["max_announcements_per_cycle"]:
            path_msg = self._build_path_message(path_status, now)
            if path_msg is not None:
                messages.append(path_msg)

        # Stair/hole hazard message (autopilot only -- new in v5).
        # stair_hole_hazard is either None ("say nothing": below
        # confidence, "normal" prediction, or not a stair/hole class) or
        # (hazard_key, raw_label, confidence) from StairHoleClassifier.
        if stair_hole_hazard is not None and len(messages) < CONFIG["max_announcements_per_cycle"]:
            sp_msg = self._build_stair_hole_message(stair_hole_hazard, now)
            if sp_msg is not None:
                messages.append(sp_msg)

        return messages

    def _build_path_message(self, path_status, now):
        # Only fires for continuous (glasses/webcam) mode, and only when
        # something is actually wrong -- a flat/clear path stays silent here
        # so the loop doesn't repeat "road is flat" every capture cycle.
        # (run_image_to_audio's one-shot report always states the clear
        # case too, via build_path_phrase_vi(..., include_clear=True).)
        phrase = build_path_phrase_vi(path_status, include_clear=False)
        if phrase is None:
            return None
        key = "path_blocked"
        if now - self.path_cooldowns.get(key, -1e9) < CONFIG["path_cooldown_s"]:
            return None
        self.path_cooldowns[key] = now
        return {"text": phrase + ".", "priority": -10.0, "key": key, "warning": "WARNING"}

    def _build_stair_hole_message(self, hazard, now):
        """hazard is (hazard_key, raw_label, confidence) or None. Only
        speaks when the classifier was confident AND recognized the class
        as stair/hole (that gating already happened inside
        StairHoleClassifier.classify) -- here we only add the per-hazard
        cooldown so it doesn't repeat every single capture cycle."""
        if hazard is None:
            return None
        hazard_key, raw_label, confidence = hazard
        phrase = STAIR_HOLE_PHRASE_VI.get(hazard_key)
        if phrase is None:
            return None
        key = f"stair_hole:{hazard_key}"
        if now - self.path_cooldowns.get(key, -1e9) < CONFIG["stair_hole_cooldown_s"]:
            return None
        self.path_cooldowns[key] = now
        return {
            "text": _capitalize_vi(phrase) + ".", "priority": -20.0, "key": key, "warning": "WARNING",
            "raw_label": raw_label, "confidence": confidence,
        }

    @staticmethod
    def _format_clause(track, distance, closing_speed):
        """The part of the announcement -- reusable both standalone
        (per-message) and combined into one sentence with other clauses
        (see build_audio_text_vi). Side-first, no warning-word prefix and
        no height clause, to match the requested concise style:
        'phía trước có xe hơi cách 3 mét'."""
        cls = CLASS_VI.get(track.class_name, track.class_name)
        side = BROAD_SIDE_VI[track.broad_side]
        dist = format_distance_vi(distance)
        approaching = closing_speed > CONFIG["approaching_speed_mps"]
        parts = [side, "có", cls, f"cách {dist}"]
        if approaching:
            parts.append(APPROACHING_VI)
        return " ".join(p for p in parts if p)


# ============================================================================
# 11b. COMBINED AUDIO SENTENCE  (used by run_image_to_audio, section 15)
# ----------------------------------------------------------------------------
# Turns the list of per-object messages + the raw path_status into ONE
# natural spoken sentence, no repeated "CHÚ Ý"/"NGUY HIỂM" per item.
# Matches the requested style:
#   "Phía trước có xe hơi cách 3 mét, bên phải có người cách 1 mét.
#    Địa hình chênh vênh bên trái và trước mặt."
# Only "critical" and (close-enough) "important" tier classes ever reach
# this stage (see CLASS_PRIORITY_VI + build_messages); "low" tier objects
# (bags, bottles, static street furniture, ...) are filtered out earlier
# so they never clutter the audio. Only the closest object per side is
# spoken, capped at `max_items` sides total.
#
# NOTE: run_on_glasses (autopilot) does NOT use this function -- it speaks
# each message from build_messages() individually through SpeechQueue, so
# the stair/hole message reaches speech there without going through this
# function at all. run_image_to_audio DOES use this function to build the
# one combined sentence it hands to TTS, so this function must explicitly
# pull the stair/hole message out of `messages` itself -- it used to only
# look at crowd/object/path entries and silently dropped the stair/hole
# message even though it was correctly present in `messages`.
# ============================================================================

def build_audio_text_vi(messages, path_status, max_items=3):
    """Composes the final spoken sentence:
        - crowd alert (if any) stays its own sentence, still says NGUY HIỂM
          -- it's an always-critical case.
        - object warnings: grouped by side, only the single closest object
          per side is spoken (so two people tracked 'ahead' don't produce
          two near-duplicate clauses), no repeated "CHÚ Ý" per item --
          e.g. "Phía trước có xe hơi cách 3 mét, bên phải có người cách 1
          mét." Objects were already filtered to critical/important-tier
          by AnnouncementManager.build_messages before reaching here.
        - path/terrain sentence next, using build_path_phrase_vi.
        - stair/hole hazard sentence(s) last (e.g. "Có bậc thang." /
          "Có hố."). Picked out of `messages` by key prefix "stair_hole:" --
          AnnouncementManager already did all the gating (confidence
          threshold, recognized class, per-hazard cooldown) before this
          message ever landed in `messages`, so if it's here it's meant
          to be spoken; this function just needs to not drop it.
    """
    crowd = next((m for m in messages if m.get("key") == "CROWD"), None)
    object_msgs = [m for m in messages if "clause" in m]
    stair_hole_msgs = [m for m in messages if m.get("key", "").startswith("stair_hole:")]

    best_per_side = {}
    for m in object_msgs:
        side = m["broad_side"]
        if side not in best_per_side or m["priority"] < best_per_side[side]["priority"]:
            best_per_side[side] = m
    grouped = sorted(best_per_side.values(), key=lambda m: m["priority"])[:max_items]

    sentences = []
    if crowd is not None:
        sentences.append(crowd["text"] + ".")
    if grouped:
        clauses = [m["clause"] for m in grouped]
        clauses[0] = _capitalize_vi(clauses[0])
        sentences.append(", ".join(clauses) + ".")

    path_phrase = build_path_phrase_vi(path_status, include_clear=True)
    if path_phrase is not None:
        sentences.append(path_phrase + ".")

    for m in stair_hole_msgs:
        sentences.append(m["text"])  # already ends with "." -- see _build_stair_hole_message

    if not sentences:
        return "Đường phía trước có vẻ an toàn."
    return " ".join(sentences)


# ============================================================================
# 12. VIETNAMESE TTS  (Piper -- see docs/API_PYTHON.md in OHF-Voice/piper1-gpl)
# ============================================================================

class PiperTTSVi:
    def __init__(self, voice_path, use_cuda=False):
        from piper import PiperVoice
        self.voice = PiperVoice.load(voice_path, use_cuda=use_cuda)

    def speak_to_wav(self, text, out_wav_path):
        with wave.open(out_wav_path, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)
        return out_wav_path

    def speak_messages_to_wav(self, messages, out_wav_path, path_status=None):
        """path_status=None -> legacy behavior (just join message texts,
        used by continuous/glasses mode where messages already carry the
        path warning if there is one). path_status=<the analyze() result>
        -> compose the single combined sentence (used by run_image_to_audio)."""
        if path_status is not None:
            text = build_audio_text_vi(messages, path_status)
        elif not messages:
            text = "Đường phía trước có vẻ an toàn."
        else:
            text = ". ".join(m["text"] for m in messages) + "."
        return self.speak_to_wav(text, out_wav_path)


# ============================================================================
# 13. PER-FRAME PROCESSING
# ============================================================================

def process_frame(image, detector, depth_estimator, ground_plane, tracker,
                   announcer, timestamp, path_advisor=None,
                   stair_hole_classifier=None):
    """stair_hole_classifier is optional and defaults to None so this
    function's behavior is unchanged for run_image_to_audio (which does
    not pass one by default). run_on_glasses (autopilot) passes an
    instance of StairHoleClassifier -- see section 16."""
    height, width = image.shape[:2]

    # ---- Depth (raw, unitless, affine-invariant) ----
    raw_depth = depth_estimator.estimate_raw(image)

    # ---- Detection ----
    raw_detections = detector.detect(image, CONFIG["detector_conf"])
    prelim_detections = []
    for det in raw_detections:
        x1, y1, x2, y2 = det["bbox_raw"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width - 1, int(x2)), min(height - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        prelim_detections.append({**det, "bbox": (x1, y1, x2, y2)})

    # ---- Ground-plane calibration (root-cause fix lives here) ----
    expected_rows = ground_plane.expected_depth_rows(height, width)
    calib = ground_plane.calibrate(raw_depth, prelim_detections, width, height)

    if calib is not None:
        a, b, is_stale = calib
        metric_depth = GroundPlaneModel.to_metric(raw_depth, a, b)
    else:
        metric_depth = None  # fall back to per-object geometric proxy below

    # ---- Finish building detections (distance, zones) ----
    detections = []
    for det in prelim_detections:
        x1, y1, x2, y2 = det["bbox"]
        center_x = (x1 + x2) / 2
        position, _ = get_horizontal_position(center_x, width)
        height_zone = get_height_zone(y1, y2, height)

        distance = None
        if metric_depth is not None:
            distance = robust_object_distance(metric_depth, x1, y1, x2, y2, height_zone, width, height)
        if distance is None:
            closeness = geometric_distance_proxy(x1, y1, x2, y2, width, height)
            distance = CONFIG["max_warning_m"] * (1.0 - closeness) + 0.3

        max_cutoff = CONFIG["overhead_max_m"] if height_zone == "HEAD-LEVEL" else CONFIG["max_warning_m"]
        if distance > max_cutoff:
            continue

        detections.append({
            "class": det["class"], "confidence": det["confidence"], "bbox": (x1, y1, x2, y2),
            "position": position, "broad_side": BROAD_SIDE[position],
            "height_zone": height_zone, "distance_m": distance,
        })

    tracks = tracker.update(detections, width, height, timestamp)

    path_status = None
    if path_advisor is not None and metric_depth is not None:
        path_status = path_advisor.analyze(metric_depth, expected_rows, width, height)

    # ---- Stair / hole hazard classification (autopilot only) ----
    stair_hole_hazard = None
    if stair_hole_classifier is not None:
        stair_hole_hazard = stair_hole_classifier.classify(image)

    processing_elapsed = time.time() - timestamp
    extra_latency_s = processing_elapsed + CONFIG["tts_onset_latency_s"]
    messages = announcer.build_messages(
        tracks, timestamp, extra_latency_s=extra_latency_s,
        path_status=path_status, stair_hole_hazard=stair_hole_hazard,
    )

    return metric_depth, detections, tracks, messages, path_status, stair_hole_hazard


# ============================================================================
# 14. BUILD PIPELINE
# ============================================================================

def build_stair_hole_classifier():
    """Shared constructor for StairHoleClassifier so both entry points
    (run_image_to_audio and run_on_glasses) build it the same way from
    CONFIG, instead of duplicating the argument list in two places."""
    return StairHoleClassifier(
        model_path=CONFIG["stair_hole_model_path"],
        imgsz=CONFIG["stair_hole_imgsz"],
        conf_threshold=CONFIG["stair_hole_conf_threshold"],
        ignore_labels=CONFIG["stair_hole_ignore_labels"],
        device=CONFIG["zipdepth_device"],  # reuse the same cpu/cuda choice as the rest of the pipeline
    )


def build_pipeline():
    """Builds the core pipeline shared by both entry points."""
    detector = Yolo26Detector(CONFIG["detector_model_path"], KEEP_CLASSES)
    depth_estimator = ZipDepthEstimator(
        checkpoint_path=CONFIG["zipdepth_checkpoint"],
        variant=CONFIG["zipdepth_variant"],
        device=CONFIG["zipdepth_device"],
        input_size=CONFIG["zipdepth_input_size"],
        ensure_multiple_of=CONFIG["zipdepth_ensure_multiple_of"],
        use_npu_upsample=CONFIG["zipdepth_use_npu_upsample"],
    )
    ground_plane = GroundPlaneModel(
        hfov_deg=CONFIG["horizontal_fov_deg"],
        camera_height_m=CONFIG["camera_height_m"],
        camera_pitch_deg=CONFIG["camera_pitch_deg"],
    )
    tracker = ObjectTracker()
    announcer = AnnouncementManager()
    path_advisor = FreeSpacePathAdvisor()
    return detector, depth_estimator, ground_plane, tracker, announcer, path_advisor


# ============================================================================
# 15. SINGLE IMAGE -> VIETNAMESE WARNING -> AUDIO  (Function 1 -- unchanged)
# ============================================================================

def run_image_to_audio(image_path, out_wav_path="out.wav", out_image_path=None, verbose=True,
                        run_stair_hole=True):
    """Single-shot path. run_stair_hole=True (default) now also runs
    StairHoleClassifier on the image so you can evaluate it from a
    single test photo -- classify() prints its raw label+confidence
    every call (see verbose_print on StairHoleClassifier), and the
    result is still only added to the spoken warning if it clears
    stair_hole_conf_threshold and is a recognized stair/hole class
    (a "normal" prediction never gets added, regardless of confidence);
    otherwise it stays out of `messages` exactly like in autopilot mode.
    Pass run_stair_hole=False to skip loading that model entirely."""
    detector, depth_estimator, ground_plane, tracker, announcer, path_advisor = build_pipeline()
    print(f"[DEBUG] run_stair_hole={run_stair_hole} -- about to build classifier..." if run_stair_hole
          else "[DEBUG] run_stair_hole=False -- classifier will NOT run")
    stair_hole_classifier = build_stair_hole_classifier() if run_stair_hole else None
    if stair_hole_classifier is not None:
        print(f"[DEBUG] classifier loaded from: {CONFIG['stair_hole_model_path']}")

    CONFIG["confirm_frames"] = 1  # single-shot: don't wait for a 2nd frame to "confirm" a track
    CONFIG["min_gap_between_any_announcement_s"] = 0.0
    CONFIG["max_announcements_per_cycle"] = 10

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    timestamp = time.time()
    metric_depth, detections, tracks, messages, path_status, stair_hole_hazard = process_frame(
        image, detector, depth_estimator, ground_plane, tracker, announcer, timestamp, path_advisor,
        stair_hole_classifier,
    )

    if verbose:
        print("=" * 70)
        print(f"DETECTIONS: {len(detections)}   |   MESSAGES: {len(messages)}")
        print("=" * 70)
        for i, m in enumerate(messages):
            print(f"{i + 1}. {m['text']}")
        if path_status is not None:
            print("\nWALKABLE PATH:")
            for name, (status, near_depth) in zip(["LEFT", "CENTER", "RIGHT"], path_status):
                extra = f" (near={near_depth:.2f}m)" if near_depth is not None else ""
                print(f"  {name}: {status}{extra}")
        else:
            print(
                "\nWALKABLE PATH: unavailable -- ground-plane depth calibration "
                "failed on this frame (common on noisy/low-light/reflective "
                "scenes with too little reliable ground-band signal). No "
                "prior-frame fallback exists for a single standalone image, "
                "so terrain status is reported as unknown rather than guessed."
            )
        if stair_hole_hazard is not None:
            hazard_key, raw_label, confidence = stair_hole_hazard
            print(f"\nSTAIR/HOLE: {hazard_key} (label={raw_label!r}, conf={confidence:.3f}) -- IN WARNING")
        elif run_stair_hole:
            print("\nSTAIR/HOLE: no hazard above threshold this frame (see per-call log above)")

    audio_text = build_audio_text_vi(messages, path_status)
    tts = PiperTTSVi(CONFIG["piper_voice_path"])
    tts.speak_to_wav(audio_text, out_wav_path)

    if verbose:
        print(f"\nAUDIO TEXT: {audio_text}")
        print(f"AUDIO SAVED TO: {out_wav_path}")

    if out_image_path:
        annotated = draw_navigation_output(image.copy(), tracks, path_status)
        cv2.imwrite(out_image_path, annotated)

    return messages, path_status, out_wav_path, audio_text


# ============================================================================
# 16. CONTINUOUS VIDEO / GLASSES LOOPS  (Function 2 = run_on_glasses = "autopilot")
# ============================================================================

class SpeechQueue:
    def __init__(self, speak_fn, stop_fn=None, chars_per_second=None):
        self.speak_fn = speak_fn
        self.stop_fn = stop_fn
        self.chars_per_second = chars_per_second or CONFIG["tts_chars_per_second"]

    def _duration(self, text):
        return len(text) / self.chars_per_second

    def enqueue(self, messages):
        if not messages:
            return
        spoken_time, interrupted = 0.0, False
        for m in messages:
            dur = self._duration(m["text"])
            if m["warning"] == "DANGER" and not interrupted and self.stop_fn is not None:
                self.stop_fn()
                interrupted = True
            if spoken_time + dur > CONFIG["max_speech_queue_s"] and m["warning"] != "DANGER":
                break
            self.speak_fn(m["text"])
            spoken_time += dur


class CaptureScheduler:
    def __init__(self):
        self._alert_cycles_left = 0

    def next_interval(self, messages):
        levels = {m["warning"] for m in messages}
        if "DANGER" in levels:
            self._alert_cycles_left = CONFIG["alert_hold_cycles"]
        elif self._alert_cycles_left > 0:
            self._alert_cycles_left -= 1
        if self._alert_cycles_left > 0 or "WARNING" in levels:
            return CONFIG["capture_interval_s"]["alert"]
        return CONFIG["capture_interval_s"]["calm"]


def make_piper_speak_fn(voice_path):
    """speak_fn for run_on_glasses(): synthesizes + plays each message live."""
    tts = PiperTTSVi(voice_path)
    import tempfile, os
    try:
        import simpleaudio as sa
        can_play = True
    except ImportError:
        can_play = False

    def _speak(text):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        tts.speak_to_wav(text, tmp_path)
        if can_play:
            sa.WaveObject.from_wave_file(tmp_path).play().wait_done()
        os.remove(tmp_path)

    return _speak


def run_on_glasses(capture_fn, speak_fn, stop_fn=None, max_cycles=None):
    """AUTOPILOT / continuous loop. This is the only place
    StairHoleClassifier is built and used, per your request -- it is
    NOT used by run_image_to_audio (section 15) unless you pass
    run_stair_hole=True there too."""
    detector, depth_estimator, ground_plane, tracker, announcer, path_advisor = build_pipeline()
    stair_hole_classifier = build_stair_hole_classifier()

    scheduler = CaptureScheduler()
    speech_queue = SpeechQueue(speak_fn, stop_fn)
    cycles = 0
    while True:
        frame = capture_fn()
        messages = []
        if frame is not None:
            timestamp = time.time()
            _, _, _, messages, _, _ = process_frame(
                frame, detector, depth_estimator, ground_plane, tracker, announcer, timestamp,
                path_advisor, stair_hole_classifier,
            )
            speech_queue.enqueue(messages)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        time.sleep(scheduler.next_interval(messages))


# ============================================================================
# 17. MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blind navigation: image -> Vietnamese warning -> audio")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--audio", default="out.wav", help="Output .wav path")
    parser.add_argument("--annotated", default=None, help="Optional: save an annotated debug image here")
    # Paths differ per machine -- override instead of hand-editing CONFIG.
    parser.add_argument("--checkpoint", default=None, help="Override CONFIG['zipdepth_checkpoint']")
    parser.add_argument("--voice", default=None, help="Override CONFIG['piper_voice_path'] (vi_VN .onnx)")
    parser.add_argument("--yolo", default=None, help="Override CONFIG['detector_model_path']")
    parser.add_argument("--stair-hole-model", default=None, help="Override CONFIG['stair_hole_model_path']")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override CONFIG['zipdepth_device']")
    parser.add_argument("--no-stair-hole", action="store_true", help="Skip loading the stair/hole classifier")
    args = parser.parse_args()

    if args.checkpoint:
        CONFIG["zipdepth_checkpoint"] = args.checkpoint
    if args.voice:
        CONFIG["piper_voice_path"] = args.voice
    if args.yolo:
        CONFIG["detector_model_path"] = args.yolo
    if args.stair_hole_model:
        CONFIG["stair_hole_model_path"] = args.stair_hole_model
    if args.device:
        CONFIG["zipdepth_device"] = args.device

    run_image_to_audio(
        args.image, out_wav_path=args.audio, out_image_path=args.annotated,
        run_stair_hole=not args.no_stair_hole,
    )