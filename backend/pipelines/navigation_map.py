"""
function3_map.py — map-guided walking navigation with a unified
nav + hazard spoken phrase.

WHY THIS FILE LOOKS THE WAY IT DOES
----------------------------------------------------------------------------
PROBLEM 1 — false reroutes:
  `_distance_to_route_m` used to measure distance to the nearest polyline
  VERTEX. Mapbox `overview=full` geometry is still sparse on long straight
  roads, so a walker who is ON the path (mid-segment) can sit 30-40 m from
  the nearest vertex and trip `OFF_ROUTE_M` (25 m) every block. The logs
  ("Bạn đã đi lệch đường" looping, walker reset to step 1) are that bug.

  FIX: true point-to-segment distance on a local equirectangular plane.

PROBLEM 2 — fragmented audio:
  HazardWorker and RouteGuide used to push independent clips into the
  priority queue, so the user heard a turn, then a hazard, then a stale
  "Bắt đầu di chuyển" after a false reroute -- never one phrase that
  combined live remaining distance with what was in front of them.

  FIX: `CombinedGuidanceLoop` runs one cycle every `cycle_interval_s`
  (default 10 s):
    1. sample GPS, recompute remaining along-track distance
    2. run Function 2 hazard detection
    3. speak ONE utterance: [live nav] + [hazard(s)]
       e.g. "Đi 240 mét rồi rẽ phải. Chú ý phía trước có hố sâu."
  (nav first, then hazard(s) -- see compose_combined_utterance)

PROBLEM 3 — hazards silently vanish on the 10s combined cadence:
  Function2_autopilot's tracker requires an object to be seen in
  `confirm_frames` (default 2) separate detection cycles before it's
  "confirmed" and allowed into spoken output, and drops any track older
  than `max_track_match_time_s` (default 6.0s). CombinedGuidanceLoop only
  detects once per `cycle_interval_s` (10s default) -- longer than the
  6s track lifetime -- so a track created one cycle ALWAYS times out
  before the next cycle can re-see it. It can never reach 2 sightings,
  confirmed never becomes True, and every real hazard is silently
  filtered out of `messages` every single cycle, even with an object
  clearly in frame.

  FIX: CombinedGuidanceLoop.__init__ forces `confirm_frames = 1` and
  `min_gap_between_any_announcement_s = 0.0` on its own pipeline, the
  same override run_image_to_audio already applies for its single-frame
  case -- each combined cycle IS one independent snapshot, not a
  continuous stream, so it should confirm on first sighting.

PROBLEM 4 — brittle access to Function2_autopilot's config:
  Earlier versions of this file indexed `pl.CONFIG[...]` directly. If
  Function2_autopilot's tunables dict is renamed, moved, or only built
  by a function call, that raises a bare
  `AttributeError: module 'Function2_autopilot' has no attribute 'CONFIG'`
  at import/call time with no hint about what changed.

  FIX: `_pl_config()` below resolves and caches the dict once, checking
  a few common names (`CONFIG`, `config`, `SETTINGS`, `settings`) and a
  `get_config()` factory function, and raises a clear, actionable error
  if none of those exist. Every place in this file that used to say
  `pl.CONFIG[...]` now goes through `_pl_config()[...]`. If your actual
  Function2_autopilot.py exposes its tunables under some other name or
  shape, add it to the `attr` tuple in `_pl_config()`.

SAFETY (not abandoned):
  Mapbox HTTP is still NOT on the detection hot path -- it only runs when
  a destination is set or a TRUE off-route reroute fires. TTS is still a
  non-blocking push into `PriorityAudioQueue`. A DANGER in the combined
  phrase uses HAZARD_DANGER so it interrupts whatever is currently playing.
  `HazardWorker` remains available if you ever want a fully independent,
  lower-latency hazard stream (see its class docstring); the default
  session/live-test path uses CombinedGuidanceLoop.

INTEGRATION POINTS (injected callables, nothing here assumes a stack):
  - capture_fn()            -> next camera frame (np.ndarray, BGR) or None
  - get_location_fn()       -> (lon, lat) tuple from the phone's GPS, or None
  - bluetooth_audio_fn()    -> path to a newly-received destination audio
                                clip from the ESP32-S3, or None if nothing new

Requires Function2_autopilot.py and Function0_voice_control.py importable
from the same directory (or PYTHONPATH).
"""

import heapq
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

from pipelines import autopilot as pl
from pipelines import voice_control as fv0


EARTH_RADIUS_M = 6371000.0


# ============================================================================
# 0. SAFE ACCESS TO Function2_autopilot's CONFIG
# ============================================================================

def _pl_config():
    """Function2_autopilot's tunable-settings dict, resolved lazily and
    cached.

    Tolerant of the dict living under a different name across
    Function2_autopilot versions, so a rename/move there doesn't crash
    Function3 with a bare `AttributeError: module 'Function2_autopilot'
    has no attribute 'CONFIG'` -- it fails once, with a message telling
    you exactly what to fix, instead of at a random call site.

    If your Function2_autopilot.py exposes its tunables under some other
    name (or as something other than a plain dict), add that name to the
    `attr` tuple below, or adjust this function to match how it's built.
    """
    if not hasattr(_pl_config, "_cache"):
        cfg = None
        for attr in ("CONFIG", "config", "SETTINGS", "settings"):
            candidate = getattr(pl, attr, None)
            if isinstance(candidate, dict):
                cfg = candidate
                break
        if cfg is None:
            get_cfg = getattr(pl, "get_config", None)
            if callable(get_cfg):
                cfg = get_cfg()
        if cfg is None:
            raise AttributeError(
                "Function2_autopilot has no tunables dict under any of "
                "CONFIG/config/SETTINGS/settings, and has no get_config() "
                "function either. Open Function2_autopilot.py, find the "
                "actual name/shape of its settings object (e.g. it may "
                "live on a class, or only be built inside build_pipeline()), "
                "and update _pl_config() in Function3_map.py to match."
            )
        _pl_config._cache = cfg
    return _pl_config._cache


# ============================================================================
# 1. PRIORITY LEVELS
# ============================================================================

class Priority(IntEnum):
    """Lower value = spoken first / allowed to interrupt. Hazard tiers come
    from Function2_autopilot's own DANGER/WARNING levels; map tiers are always
    below both, per the brief ("real-time hazard detection alerts
    immediately override or take priority over standard map instructions")."""
    HAZARD_DANGER = 0
    HAZARD_WARNING = 1
    MAP_CRITICAL = 2   # arrival, "destination not found", reroute notices
    MAP_STEP = 3        # ordinary turn-by-turn instructions


HAZARD_WARNING_LEVEL_TO_PRIORITY = {
    "DANGER": Priority.HAZARD_DANGER,
    "WARNING": Priority.HAZARD_WARNING,
}

# Spoken prefix so a hazard mixed into the combined phrase is unmistakably a
# hazard and not part of the nav instruction -- e.g. "Đi 240 mét rồi rẽ
# phải." + "Chú ý phía trước có hố sâu." Function2_autopilot's own per-object
# AnnouncementManager output stays prefix-free (that's used standalone,
# without map audio around it); this prefix is added only here, at the
# point where hazard and nav text are about to share one utterance.
HAZARD_PREFIX_VI = {"DANGER": "Nguy hiểm, ", "WARNING": "Chú ý, "}


def _with_hazard_prefix(text, warning_level):
    prefix = HAZARD_PREFIX_VI.get(warning_level)
    if not prefix or not text:
        return text
    return prefix + text[0].lower() + text[1:]


def _format_distance_vi(meters):
    """Prefer Function 2's formatter; fall back to a simple Vietnamese form."""
    try:
        return pl.format_distance_vi(meters)
    except Exception:
        meters = max(float(meters), 1.0)
        if meters >= 1000:
            km = meters / 1000.0
            return f"{km:.1f}".replace(".", ",") + " ki-lô-mét"
        return f"{int(round(meters))} mét"


def _capitalize_vi(text):
    try:
        return pl._capitalize_vi(text)
    except Exception:
        if not text:
            return text
        return text[0].upper() + text[1:]


def _distance_of_message(m):
    """Best-effort distance (metres) for a hazard message dict, used only
    to pick the nearest of several duplicate sightings. Prefers an
    explicit numeric field if Function2 provides one; otherwise parses the
    number immediately before "mét" out of the Vietnamese text itself
    (e.g. "... cách 3 mét." -> 3.0). Unparseable messages sort last."""
    for key in ("distance_m", "distance", "dist_m"):
        d = m.get(key)
        if isinstance(d, (int, float)):
            return float(d)
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*mét", m.get("text") or "")
    if match:
        return float(match.group(1).replace(",", "."))
    return float("inf")


def _dedupe_hazard_messages(messages):
    """Collapse duplicate/near-duplicate hazard sightings to one per
    (label, side, warning) group, keeping the nearest.

    Function2's own single-shot path (run_image_to_audio) already
    collapses several tracked sightings of the same hazard down to one
    representative line before building its final spoken text -- that's
    why running Function 2 alone on a photo full of one motorcycle
    tracked 4 times still says it only once. CombinedGuidanceLoop instead
    calls Function2's lower-level process_frame() directly per cycle and
    got the RAW per-track messages, so the same motorcycle could appear
    3-4 times in one combined phrase ("...xe máy cách 2 mét... cách 3
    mét... cách 3 mét... cách 4 mét."). This groups messages by
    (class label, side, warning level) and keeps only the nearest one per
    group -- what a listener actually needs is the closest instance of
    each hazard, not every track sighting of it.
    """
    if not messages:
        return messages
    best = {}
    order = []
    for m in messages:
        key = (
            m.get("class_name") or m.get("raw_label") or m.get("key"),
            m.get("broad_side"),
            m.get("warning"),
        )
        if key not in best:
            order.append(key)
            best[key] = m
        elif _distance_of_message(m) < _distance_of_message(best[key]):
            best[key] = m
    return [best[k] for k in order]


def _format_path_status_vi(path_status):
    """Best-effort Vietnamese phrase for the walkable-path status
    Function2's process_frame() returns (the same status printed by
    Function2 as e.g. 'LEFT: CLEAR (near=2.00m)' / 'CENTER: CLEAR ...' /
    'RIGHT: CLEAR ...', and folded into its own 'AUDIO TEXT' as "Đường
    bằng phẳng." when every side is clear).

    Tries, in order: a formatter Function2 exposes itself; a dict/object
    with LEFT/CENTER/RIGHT entries each carrying a 'status' field. If the
    shape doesn't match, returns None rather than guessing wrong -- a
    missing "path is clear" sentence is far less harmful than a wrong
    one, and every caller here already treats None as "say nothing extra".
    If this doesn't produce "Đường bằng phẳng." on a frame you know is
    clear, tell me the actual shape of the 5th value process_frame()
    returns and I'll fix this function specifically instead of guessing
    again.
    """
    if path_status is None:
        return None
    try:
        return pl.format_path_status_vi(path_status)
    except Exception:
        pass
    try:
        if isinstance(path_status, dict):
            sides = path_status
        else:
            sides = {
                name: getattr(path_status, name)
                for name in ("LEFT", "CENTER", "RIGHT")
                if hasattr(path_status, name)
            }
        if not sides:
            return None
        statuses = []
        for v in sides.values():
            s = v.get("status") if isinstance(v, dict) else getattr(v, "status", None)
            if s is None:
                return None
            statuses.append(str(s).upper())
        if statuses and all(s == "CLEAR" for s in statuses):
            return "Đường bằng phẳng."
        return None
    except Exception:
        return None


def compose_combined_utterance(hazard_messages, nav_text, path_status_text=None):
    """Merge live nav + hazard(s) + path status into a single spoken phrase.

    Order is [nav] then [hazard...] then [path status] -- the user hears
    where they're going first, then what's in front of them right now,
    then whether the ground itself is clear, matching Function2's own
    single-shot phrasing style. Multiple hazards sort DANGER before
    WARNING and are deduped to one per (label, side, warning) group
    before this is called (see _dedupe_hazard_messages). `path_status_text`
    -- e.g. "Đường bằng phẳng." -- is appended unprefixed, the same way
    Function2's own AUDIO TEXT states it plainly rather than as a "Chú ý"
    hazard. Returns None if there is nothing to say.

    Example:
      "Đi 5 mét rồi rẽ trái. Bên phải có xe máy cách 2 mét. Đường bằng phẳng."
    """
    sentences = []

    nav = (nav_text or "").strip().rstrip(" .")
    if nav:
        sentences.append(nav)

    ordered = sorted(
        [m for m in (hazard_messages or []) if m.get("text")],
        key=lambda m: 0 if m.get("warning") == "DANGER" else 1,
    )
    for m in ordered:
        prefixed = _with_hazard_prefix(m["text"], m.get("warning"))
        if prefixed:
            sentences.append(prefixed.rstrip(" ."))

    path_txt = (path_status_text or "").strip().rstrip(" .")
    if path_txt:
        sentences.append(path_txt)

    if not sentences:
        return None
    text = ". ".join(sentences) + "."
    return text[0].upper() + text[1:]


# ============================================================================
# 2. PRIORITY AUDIO QUEUE + PLAYER
# ============================================================================

class PriorityAudioQueue:
    """Thread-safe min-priority queue. put() never blocks; get() blocks
    until an item is available or the queue is closed."""

    def __init__(self):
        self._heap = []
        self._counter = itertools.count()   # FIFO tie-breaker within a priority
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

    def put(self, priority, text, meta=None):
        with self._not_empty:
            heapq.heappush(self._heap, (int(priority), next(self._counter), text, meta or {}))
            self._not_empty.notify()

    def get(self):
        """Returns (priority, text, meta), or None once closed with nothing left."""
        with self._not_empty:
            while not self._heap and not self._closed:
                self._not_empty.wait()
            if not self._heap:
                return None
            priority, _, text, meta = heapq.heappop(self._heap)
            return priority, text, meta

    def close(self):
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()


class AudioPlayerThread(threading.Thread):
    """Single consumer of the priority queue. Synthesizes each message with
    Piper to a real .wav file, saves it, and plays it live through
    whichever backend is available; a HAZARD_DANGER arrival can call
    interrupt_for_hazard() to kill whatever's currently playing (map
    instruction or a lower-priority hazard) so it never delays the danger
    alert.

    AUDIO BACKENDS (tried in this order, whichever is first available):
      1. `simpleaudio` (Python lib) if installed and it can open a device.
      2. `paplay` (PulseAudio CLI) if on PATH.
      3. `aplay` (ALSA CLI) if on PATH.
      4. `ffplay` (ffmpeg CLI, headless) if on PATH.
    If none work, the .wav is still saved and the text still printed --
    playback failing is never allowed to mean "the message is lost".

    RELIABILITY: this is the ONLY consumer of the queue. If it ever raises
    out of run(), every message pushed afterward is silently stranded
    forever. So:
      - run() never lets an exception from _speak() escape.
      - _speak() always prints the text to the console BEFORE attempting
        synthesis or playback.
      - synthesis failures and playback failures are caught and reported
        separately, so you can tell which stage broke.
    """

    _FALLBACK_PLAYERS = (
        ("paplay", ["paplay", "{path}"]),
        ("aplay", ["aplay", "-q", "{path}"]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "{path}"]),
    )

    def __init__(self, tts, queue_, audio_out_dir=None):
        super().__init__(daemon=True)
        self.tts = tts
        self.queue = queue_
        self._current_play_obj = None
        self._current_proc = None
        self._play_lock = threading.Lock()
        self._msg_counter = itertools.count(1)

        # Every synthesized clip is saved here permanently (real, listenable
        # audio files) in addition to whatever live playback does.
        self.audio_out_dir = Path(audio_out_dir) if audio_out_dir else None
        if self.audio_out_dir:
            self.audio_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import simpleaudio as sa
            self._sa = sa
        except ImportError:
            self._sa = None

        self._fallback_cmd = None
        self._fallback_name = None
        for name, template in self._FALLBACK_PLAYERS:
            if shutil.which(name):
                self._fallback_cmd = template
                self._fallback_name = name
                break

        if self._sa is not None:
            print("[AudioPlayerThread] live playback via 'simpleaudio'.")
        elif self._fallback_cmd is not None:
            print(f"[AudioPlayerThread] 'simpleaudio' not installed -- "
                  f"falling back to '{self._fallback_name}' for live playback.")
        else:
            print("[AudioPlayerThread] no audio backend available (no 'simpleaudio', "
                  "and none of paplay/aplay/ffplay found on PATH). Messages will still "
                  "be synthesized to real .wav files"
                  + (f" under {self.audio_out_dir}" if self.audio_out_dir else "")
                  + " and printed, just not played live. Install simpleaudio, or "
                  "paplay/aplay/ffplay, to hear them as they happen.")

    def interrupt_for_hazard(self):
        with self._play_lock:
            if self._current_play_obj is not None:
                try:
                    self._current_play_obj.stop()
                except Exception:
                    pass
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            _priority, text, _meta = item
            try:
                self._speak(text)
            except Exception as e:
                # Must never propagate: this is the queue's only consumer.
                # If run() dies here, every later message is stranded.
                print(f"[AudioPlayerThread] unexpected error speaking {text!r}: {e!r}")

    def _speak(self, text):
        # Console echo happens first and unconditionally -- this is the one
        # channel that must never fail, so you can always see what should
        # have been said even if TTS/playback is broken.
        print(f"[audio] {text}", flush=True)

        if self.audio_out_dir:
            idx = next(self._msg_counter)
            out_path = str(self.audio_out_dir / f"{idx:04d}.wav")
            owns_tmp_file = False
        else:
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out_path = f.name
            f.close()
            owns_tmp_file = True

        try:
            self.tts.speak_to_wav(text, out_path)
        except Exception as e:
            print(f"[AudioPlayerThread] TTS synthesis failed for {text!r}: {e!r}")
            if owns_tmp_file:
                self._cleanup(out_path)
            return

        if self.audio_out_dir:
            print(f"[audio] saved: {out_path}")

        self._play(out_path, text)

        if owns_tmp_file:
            self._cleanup(out_path)

    def _play(self, wav_path, text):
        if self._sa is not None:
            try:
                wave_obj = self._sa.WaveObject.from_wave_file(wav_path)
                with self._play_lock:
                    self._current_play_obj = wave_obj.play()
                play_obj = self._current_play_obj
                if play_obj is not None:
                    play_obj.wait_done()
                return
            except Exception as e:
                print(f"[AudioPlayerThread] simpleaudio playback failed for {text!r}: {e!r} "
                      "-- trying fallback player if one is available.")
            finally:
                with self._play_lock:
                    self._current_play_obj = None

        if self._fallback_cmd is not None:
            try:
                cmd = [part.format(path=wav_path) for part in self._fallback_cmd]
                with self._play_lock:
                    self._current_proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                proc = self._current_proc
                proc.wait()
                return
            except Exception as e:
                print(f"[AudioPlayerThread] '{self._fallback_name}' playback failed "
                      f"for {text!r}: {e!r}")
            finally:
                with self._play_lock:
                    self._current_proc = None
        # No backend worked (or none available) -- text was printed and the
        # .wav (if audio_out_dir is set) is still on disk, so nothing is lost.

    @staticmethod
    def _cleanup(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass


class NavigationAudioSystem:
    """Owns the one priority queue + one player thread that both
    CombinedGuidanceLoop (and, if used, HazardWorker / RouteGuide critical
    events) push into."""

    def __init__(self, piper_voice_path, audio_out_dir=None):
        self.tts = pl.PiperTTSVi(piper_voice_path)
        self._queue = PriorityAudioQueue()
        self._player = AudioPlayerThread(self.tts, self._queue, audio_out_dir=audio_out_dir)

    def start(self):
        self._player.start()

    def push(self, priority, text, meta=None):
        if priority == Priority.HAZARD_DANGER:
            self._player.interrupt_for_hazard()
        self._queue.put(priority, text, meta)

    def stop(self, join_timeout_s=5.0):
        self._queue.close()
        self._player.join(timeout=join_timeout_s)
        if self._player.is_alive():
            print("[NavigationAudioSystem] player thread did not stop within "
                  f"{join_timeout_s}s (likely mid-playback of a long clip) -- "
                  "it will keep running as a daemon thread.")


# ============================================================================
# 3. HAZARD WORKER  (Function 2, continuous, background thread -- optional,
#    independent-cadence alternative/complement to CombinedGuidanceLoop)
# ============================================================================

class HazardWorker(threading.Thread):
    """Runs Function2_autopilot's detect/depth/track/announce loop, exactly
    like run_on_glasses, but instead of speaking messages itself it pushes
    them into the shared NavigationAudioSystem the moment they're detected
    -- independent of CombinedGuidanceLoop's cycle_interval_s.

    The default session/live-test path uses CombinedGuidanceLoop alone,
    which ties hazard detection to the same cadence as nav announcements
    (see CombinedGuidanceLoop's docstring for the latency trade-off that
    implies). Run this alongside it if you need hazards spoken the instant
    they're seen rather than waiting for the next combined cycle.

    CADENCE: `min_capture_interval_s` (default 5.0) is a floor on the time
    between the END of one detect/announce cycle and the START of the
    next -- matches the ESP32-S3 sending periodic photos rather than a
    live video stream, and avoids re-announcing the same hazard many times
    a second on an unchanging scene. CaptureScheduler can still ask for a
    LONGER wait (e.g. nothing nearby); it just can never make the gap
    shorter than this floor."""

    def __init__(self, capture_fn, audio_system, max_cycles=None, min_capture_interval_s=5.0):
        super().__init__(daemon=True)
        self.capture_fn = capture_fn
        self.audio_system = audio_system
        self.max_cycles = max_cycles
        self.min_capture_interval_s = min_capture_interval_s
        self._stop_event = threading.Event()
        (self.detector, self.depth_estimator, self.ground_plane, self.tracker,
         self.announcer, self.path_advisor) = pl.build_pipeline()
        self.stair_hole_classifier = pl.build_stair_hole_classifier()
        self.scheduler = pl.CaptureScheduler()

    def stop(self):
        self._stop_event.set()

    def run(self):
        cycles = 0
        messages = []
        while not self._stop_event.is_set():
            frame = self.capture_fn()
            messages = []
            if frame is not None:
                timestamp = time.time()
                _, _, _, messages, _, _ = pl.process_frame(
                    frame, self.detector, self.depth_estimator, self.ground_plane,
                    self.tracker, self.announcer, timestamp, self.path_advisor,
                    self.stair_hole_classifier,
                )
                messages = _dedupe_hazard_messages(messages)
                for m in messages:
                    priority = HAZARD_WARNING_LEVEL_TO_PRIORITY.get(m["warning"], Priority.HAZARD_WARNING)
                    text = _with_hazard_prefix(m["text"], m["warning"])
                    self.audio_system.push(priority, text, meta=m)
            cycles += 1
            if self.max_cycles is not None and cycles >= self.max_cycles:
                break
            scheduler_interval = self.scheduler.next_interval(messages) or 0.0
            wait_s = max(self.min_capture_interval_s, scheduler_interval)
            self._stop_event.wait(wait_s)


# ============================================================================
# 4. MAPBOX CLIENT  (Search Box API forward search + Directions API walking)
# ============================================================================

class MapboxBackendError(RuntimeError):
    pass


class MapboxClient:
    """
    mode="backend" (default, use this in production): calls YOUR OWN
    backend's /mapbox/search and /mapbox/directions/<coords> endpoints.
    Your backend holds the Mapbox SECRET token server-side and proxies
    these two calls -- the phone app / this code never sees it. This is
    the setup the brief calls for, since the phone is just the hub and the
    ESP32 only streams raw audio/photo to it.

    mode="direct" (dev/testing only, off-device): calls Mapbox directly
    using a PUBLIC/RESTRICTED token. Never ship this mode, and never pass
    a secret token to it.

    Uses exactly two Mapbox products, per the brief:
      - Search Box API, forward-search endpoint (non-interactive,
        per-request billing) -- NOT Geocoding (no POI coverage) and NOT
        the interactive suggest/retrieve pair (session billing, meant for
        a type-ahead UI box, not a single spoken utterance).
      - Directions API, walking profile -- NOT the Navigation SDK, so we
        keep full control of the (Vietnamese) audio.
    """

    def __init__(self, mode="backend", backend_base_url=None, public_token=None, timeout_s=6.0):
        self.mode = mode
        self.timeout_s = timeout_s
        if mode == "backend":
            if not backend_base_url:
                raise ValueError("backend_base_url is required in 'backend' mode")
            self.backend_base_url = backend_base_url.rstrip("/")
        elif mode == "direct":
            if not public_token:
                raise ValueError("public_token is required in 'direct' (dev-only) mode")
            self.public_token = public_token
        else:
            raise ValueError("mode must be 'backend' or 'direct'")

    def search_box_forward(self, query, proximity_lonlat=None, language="vi", country="VN", limit=1):
        """Resolve a spoken destination (e.g. 'Vạn Hạnh Mall') to
        coordinates via the Search Box API's forward endpoint. Returns a
        list of {"name", "full_address", "longitude", "latitude"}."""
        params = {"q": query, "language": language, "country": country, "limit": limit}
        if proximity_lonlat is not None:
            params["proximity"] = f"{proximity_lonlat[0]},{proximity_lonlat[1]}"

        if self.mode == "backend":
            data = self._backend_get("/mapbox/search", params)
        else:
            params["access_token"] = self.public_token
            data = self._http_get("https://api.mapbox.com/search/searchbox/v1/forward", params)

        results = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            coords = feat.get("geometry", {}).get("coordinates", [None, None])
            results.append({
                "name": props.get("name"),
                "full_address": props.get("full_address") or props.get("place_formatted"),
                "longitude": coords[0],
                "latitude": coords[1],
            })
        return results

    def directions(self, origin_lonlat, destination_lonlat, profile="walking", language="vi"):
        """Directions API, steps=true. profile picks the routing mode the
        same way Google Maps' walking/driving/transit tabs do -- Mapbox
        exposes this as a URL segment on the SAME API rather than separate
        products: 'walking' (default here -- this is a blind pedestrian
        nav app), 'driving', 'driving-traffic', or 'cycling'. We still
        build our own Vietnamese phrases from maneuver type/modifier (see
        build_step_instructions_vi) rather than trusting Mapbox's built-in
        banner/voice text, since vi coverage there isn't guaranteed.
        overview='full' (not 'simplified') so the returned route geometry
        has enough points to drive a smooth simulated-GPS walk in testing;
        the turn-by-turn steps used for spoken instructions come from
        'steps', not from this geometry, so it's unaffected either way."""
        coords = f"{origin_lonlat[0]},{origin_lonlat[1]};{destination_lonlat[0]},{destination_lonlat[1]}"
        params = {"steps": "true", "geometries": "geojson", "overview": "full", "language": language}

        if self.mode == "backend":
            data = self._backend_get(f"/mapbox/directions/{profile}/{coords}", params)
        else:
            params["access_token"] = self.public_token
            data = self._http_get(f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{coords}", params)

        routes = data.get("routes", [])
        if not routes:
            raise MapboxBackendError(f"No {profile} route found: {data.get('message', data)}")
        return routes[0]

    def directions_walking(self, origin_lonlat, destination_lonlat, language="vi"):
        """Convenience wrapper -- this app only ever wants the walking
        profile, kept as its own method so callers don't have to remember
        to pass profile='walking' every time."""
        return self.directions(origin_lonlat, destination_lonlat, profile="walking", language=language)

    def _backend_get(self, path, params):
        return self._http_get(self.backend_base_url + path, params)

    def _http_get(self, url, params):
        import urllib.parse
        import urllib.request
        qs = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{url}?{qs}", timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ============================================================================
# 5. VIETNAMESE TURN-BY-TURN PHRASING
# ============================================================================

MANEUVER_VI = {
    ("depart", None): "bắt đầu di chuyển",
    ("arrive", None): "bạn đã đến nơi",
    ("turn", "left"): "rẽ trái",
    ("turn", "right"): "rẽ phải",
    ("turn", "slight left"): "rẽ nhẹ sang trái",
    ("turn", "slight right"): "rẽ nhẹ sang phải",
    ("turn", "sharp left"): "rẽ gắt sang trái",
    ("turn", "sharp right"): "rẽ gắt sang phải",
    ("turn", "straight"): "đi thẳng",
    ("turn", "uturn"): "quay đầu",
    ("continue", None): "tiếp tục đi thẳng",
    ("merge", None): "nhập vào đường",
    ("fork", "left"): "đi theo nhánh bên trái",
    ("fork", "right"): "đi theo nhánh bên phải",
    ("roundabout", None): "đi vào vòng xuyến",
    ("end of road", "left"): "cuối đường, rẽ trái",
    ("end of road", "right"): "cuối đường, rẽ phải",
}
DEFAULT_MANEUVER_PHRASE_VI = "tiếp tục di chuyển"

# Maneuvers that are not useful as the "next" spoken action once we have
# started walking -- live_guidance skips these in favour of the next real
# turn / arrival so we don't re-announce "Bắt đầu di chuyển" every cycle.
SKIP_AS_NEXT_MANEUVER = {"depart", "continue"}


def _maneuver_phrase_vi(maneuver):
    m_type = maneuver.get("type")
    modifier = maneuver.get("modifier")
    return (
        MANEUVER_VI.get((m_type, modifier))
        or MANEUVER_VI.get((m_type, None))
        or DEFAULT_MANEUVER_PHRASE_VI
    )


def build_step_instructions_vi(route):
    """Turns a Mapbox Directions 'route' object into a flat list of
    {"text", "distance_m", "maneuver_type", "modifier", "location"} steps,
    one per maneuver, in Vietnamese, e.g. 'Đi 100 mét rồi rẽ trái.'"""
    instructions = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            m_type = maneuver.get("type")
            modifier = maneuver.get("modifier")
            distance_m = float(step.get("distance", 0.0))
            phrase = _maneuver_phrase_vi(maneuver)

            if m_type == "arrive":
                text = "Bạn đã đến nơi."
            elif m_type == "depart":
                text = _capitalize_vi(phrase) + "."
            else:
                dist_txt = _format_distance_vi(max(distance_m, 1.0))
                text = _capitalize_vi(f"đi {dist_txt} rồi {phrase}.")

            instructions.append({
                "text": text,
                "distance_m": distance_m,
                "maneuver_type": m_type,
                "modifier": modifier,
                "location": maneuver.get("location"),   # [lon, lat] or None
            })
    return instructions


def live_step_text_vi(maneuver_type, modifier, remaining_m):
    """Live remaining-distance phrase for the upcoming maneuver.

    Unlike the static text baked at route-build time, `remaining_m` shrinks
    as the user walks so each combined cycle says a smaller number.
    """
    if maneuver_type == "arrive":
        if remaining_m < 8.0:
            return "Bạn đã đến nơi."
        dist_txt = _format_distance_vi(max(remaining_m, 1.0))
        return _capitalize_vi(f"đi {dist_txt} nữa là đến nơi.")
    if maneuver_type == "depart":
        return _capitalize_vi(_maneuver_phrase_vi({"type": "depart", "modifier": None})) + "."
    phrase = _maneuver_phrase_vi({"type": maneuver_type, "modifier": modifier})
    dist_txt = _format_distance_vi(max(remaining_m, 1.0))
    return _capitalize_vi(f"đi {dist_txt} rồi {phrase}.")


# ============================================================================
# 6. ROUTE GUIDE  (progress tracking, turn announcements, rerouting)
# ============================================================================

class RouteGuide:
    """Advances through a walking route as GPS position updates.

    Spoken MAP_STEP lines are produced by CombinedGuidanceLoop via
    `live_guidance()` (remaining along-track distance + next maneuver).
    This class still:
      - tracks along-track progress
      - fires a TRUE off-route reroute (point-to-SEGMENT distance)
      - sets arrived_event
      - pushes MAP_CRITICAL for destination search / reroute notices
    """

    ANNOUNCE_LEAD_M = 15.0    # kept for callers that still inspect it
    OFF_ROUTE_M = 25.0        # farther than this from the ROUTE PATH ITSELF -> reroute.
                               # NOT the distance to the next maneuver -- being e.g. 100m
                               # from the next turn is completely normal mid-way along a
                               # long straight step; that is NOT the same as being off the
                               # path. Measured with true point-to-segment distance, never
                               # nearest-vertex, or sparse Mapbox geometry false-reroutes.
    ARRIVE_RADIUS_M = 8.0
    PASS_MANEUVER_M = 4.0     # along-track past a maneuver => it's behind us

    def __init__(self, mapbox_client, audio_system, get_location_fn, on_route_resolved=None):
        self.mapbox = mapbox_client
        self.audio_system = audio_system
        self.get_location_fn = get_location_fn
        # Called with the new route_geometry (list of (lon, lat)) every time
        # a route is resolved, INITIAL OR REROUTE. This exists so a caller
        # driving a simulated/independent position source (SimulatedGpsWalker
        # in testing; a real GPS feed doesn't need this) can stay in sync.
        # Without it, whatever's tracking position keeps following the OLD
        # route after a reroute, looks perpetually "off route" against the
        # NEW route, triggers another reroute, and loops every tick forever.
        self.on_route_resolved = on_route_resolved
        self.instructions = []
        self.route_geometry = []   # [(lon, lat), ...] from the last resolved route -- used
                                    # by SimulatedGpsWalker in testing, ignored in production
                                    # once a real GPS feed is wired into get_location_fn
        self._cum_dist = [0.0]
        self._next_idx = 0
        self._announced_current = False
        self._destination = None
        self._lock = threading.Lock()
        self.arrived_event = threading.Event()   # set once "arrive" is announced --
                                                   # lets a caller stop the session loop
                                                   # ("... or destination is reached")

    # ----- geometry: local equirectangular + point-to-segment ---------------

    @staticmethod
    def _haversine_m(lonlat1, lonlat2):
        lon1, lat1 = lonlat1
        lon2, lat2 = lonlat2
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
        return float(2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a))))

    @staticmethod
    def _to_xy_m(lonlat, origin):
        """East/north metres of `lonlat` relative to `origin` (equirectangular)."""
        lon, lat = lonlat
        olon, olat = origin
        x = math.radians(lon - olon) * EARTH_RADIUS_M * math.cos(math.radians(olat))
        y = math.radians(lat - olat) * EARTH_RADIUS_M
        return x, y

    @classmethod
    def _closest_point_on_segment(cls, p, a, b):
        """Return (t, dist_m, snapped_lonlat) for the closest point on
        segment a→b to p. t is clamped to [0, 1]."""
        ax, ay = cls._to_xy_m(a, p)
        bx, by = cls._to_xy_m(b, p)
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            return 0.0, math.hypot(ax, ay), a
        # p is the local origin (0, 0); ap = (-ax, -ay)
        t = (-ax * abx - ay * aby) / ab2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        qx = ax + t * abx
        qy = ay + t * aby
        lon = a[0] + t * (b[0] - a[0])
        lat = a[1] + t * (b[1] - a[1])
        return t, math.hypot(qx, qy), (lon, lat)

    @classmethod
    def _polyline_cum_dist(cls, geometry):
        cum = [0.0]
        for i in range(1, len(geometry)):
            cum.append(cum[-1] + cls._haversine_m(geometry[i - 1], geometry[i]))
        return cum

    @classmethod
    def _project_onto_route(cls, loc, geometry, cum_dist=None):
        """Nearest point on the polyline.

        Returns (along_m, dist_m, snapped_lonlat). `dist_m` is true
        point-to-SEGMENT distance, not nearest-vertex.
        """
        if not geometry:
            return None, None, None
        if len(geometry) == 1:
            return 0.0, cls._haversine_m(loc, geometry[0]), geometry[0]
        if cum_dist is None:
            cum_dist = cls._polyline_cum_dist(geometry)

        best_dist = float("inf")
        best_along = 0.0
        best_pt = geometry[0]
        for i in range(len(geometry) - 1):
            t, dist, pt = cls._closest_point_on_segment(loc, geometry[i], geometry[i + 1])
            if dist < best_dist:
                seg_len = cum_dist[i + 1] - cum_dist[i]
                best_dist = dist
                best_along = cum_dist[i] + t * seg_len
                best_pt = pt
        return best_along, best_dist, best_pt

    @classmethod
    def _distance_to_route_m(cls, loc, geometry):
        """Distance from `loc` to the nearest point on any segment of the
        route polyline -- this is what "off route" should actually mean, as
        opposed to distance-to-next-maneuver OR distance-to-nearest-vertex.

        Nearest-vertex is wrong on sparse Mapbox geometry: a user walking
        on a 80 m straight between two vertices sits ~40 m from both of
        them and trips OFF_ROUTE_M even though they never left the path.
        Directions API calls in this file use overview='full', but 'full'
        is still not dense enough on long urban blocks for vertex distance
        to be a safe proxy.
        """
        _, dist, _ = cls._project_onto_route(loc, geometry)
        return dist

    def distance_to_route(self, loc=None):
        """Diagnostics helper: (dist_to_path_m, on_route) for `loc` (defaults
        to the current get_location_fn() reading), or (None, None) if there's
        no active route or no GPS fix. Used for the per-cycle debug log."""
        with self._lock:
            geometry = list(self.route_geometry)
            cum = list(self._cum_dist)
        if loc is None:
            loc = self.get_location_fn()
        if loc is None or not geometry:
            return None, None
        _, dist, _ = self._project_onto_route(loc, geometry, cum)
        return dist, (dist is not None and dist <= self.OFF_ROUTE_M)

    def _annotate_along_track(self, instructions, geometry, cum_dist):
        for inst in instructions:
            loc = inst.get("location")
            if loc is None or not geometry:
                inst["along_m"] = None
                continue
            along, _, _ = self._project_onto_route(tuple(loc), geometry, cum_dist)
            inst["along_m"] = along

    # ----- destination / routing -------------------------------------------

    def set_destination_by_name(self, destination_text_vi):
        origin = self.get_location_fn()
        if origin is None:
            self.audio_system.push(Priority.MAP_CRITICAL, "Không xác định được vị trí hiện tại.")
            return False
        try:
            hits = self.mapbox.search_box_forward(destination_text_vi, proximity_lonlat=origin)
        except Exception:
            self.audio_system.push(Priority.MAP_CRITICAL, "Không thể tìm kiếm địa điểm lúc này.")
            return False
        if not hits:
            self.audio_system.push(Priority.MAP_CRITICAL, f"Không tìm thấy {destination_text_vi}.")
            return False
        dest = hits[0]
        return self._route_to(origin, (dest["longitude"], dest["latitude"]), dest["name"])

    def _route_to(self, origin, destination, dest_name=None):
        try:
            route = self.mapbox.directions_walking(origin, destination)
        except Exception:
            self.audio_system.push(Priority.MAP_CRITICAL, "Không tìm được đường đi, vui lòng thử lại.")
            return False
        geometry = route.get("geometry", {}).get("coordinates", [])
        instructions = build_step_instructions_vi(route)
        geom = [(c[0], c[1]) for c in geometry]
        cum = self._polyline_cum_dist(geom) if geom else [0.0]
        self._annotate_along_track(instructions, geom, cum)
        with self._lock:
            self.instructions = instructions
            self.route_geometry = geom
            self._cum_dist = cum
            self._next_idx = 0
            self._announced_current = False
            self._destination = destination
        self.arrived_event.clear()

        if self.on_route_resolved:
            try:
                self.on_route_resolved(list(self.route_geometry))
            except Exception as e:
                print(f"[RouteGuide] on_route_resolved callback failed: {e!r}")

        name_txt = f" đến {dest_name}" if dest_name else ""
        self.audio_system.push(Priority.MAP_CRITICAL, f"Đã tìm thấy đường đi{name_txt}. Bắt đầu dẫn đường.")
        return True

    def _pick_next_instruction(self, along, instructions):
        """First upcoming interesting maneuver (skip depart/continue)."""
        fallback = None
        for i, inst in enumerate(instructions):
            a = inst.get("along_m")
            if a is None:
                continue
            m_type = inst.get("maneuver_type")
            passed = along is not None and a <= along + self.PASS_MANEUVER_M and m_type != "arrive"
            if passed:
                continue
            if fallback is None:
                fallback = (i, inst)
            if m_type not in SKIP_AS_NEXT_MANEUVER:
                return i, inst
        return fallback if fallback is not None else (None, None)

    def live_guidance(self):
        """Current remaining-distance instruction, recomputed from GPS.

        Returns {"text", "remaining_m", "step_index", "total_steps",
                 "maneuver_type", "dist_to_path_m"}.
        `text` is None if there is no active route.
        """
        with self._lock:
            instructions = list(self.instructions)
            geometry = list(self.route_geometry)
            cum = list(self._cum_dist)
            next_idx = self._next_idx
            arrived = self.arrived_event.is_set()

        total = len(instructions)
        empty = {
            "text": None,
            "remaining_m": None,
            "step_index": next_idx,
            "total_steps": total,
            "maneuver_type": None,
            "dist_to_path_m": None,
        }
        if arrived:
            return {
                "text": "Bạn đã đến nơi.",
                "remaining_m": 0.0,
                "step_index": max(total - 1, 0),
                "total_steps": total,
                "maneuver_type": "arrive",
                "dist_to_path_m": 0.0,
            }
        if not instructions:
            return empty

        loc = self.get_location_fn()
        if loc is None or not geometry:
            current = instructions[next_idx] if next_idx < total else None
            if current is None:
                return empty
            return {
                "text": current.get("text"),
                "remaining_m": current.get("distance_m"),
                "step_index": next_idx,
                "total_steps": total,
                "maneuver_type": current.get("maneuver_type"),
                "dist_to_path_m": None,
            }

        along, dist_path, _ = self._project_onto_route(loc, geometry, cum)
        idx, inst = self._pick_next_instruction(along, instructions)
        if inst is None:
            return {
                "text": "Bạn đã đến nơi.",
                "remaining_m": 0.0,
                "step_index": max(total - 1, 0),
                "total_steps": total,
                "maneuver_type": "arrive",
                "dist_to_path_m": dist_path,
            }

        remaining = 0.0
        if along is not None and inst.get("along_m") is not None:
            remaining = max(0.0, inst["along_m"] - along)
        elif inst.get("location") is not None:
            remaining = self._haversine_m(loc, tuple(inst["location"]))

        text = live_step_text_vi(inst.get("maneuver_type"), inst.get("modifier"), remaining)
        return {
            "text": text,
            "remaining_m": remaining,
            "step_index": idx if idx is not None else next_idx,
            "total_steps": total,
            "maneuver_type": inst.get("maneuver_type"),
            "dist_to_path_m": dist_path,
        }

    def progress(self):
        """Snapshot of guidance progress, mainly for status/debug printing:
        {"step_index", "total_steps", "next_text", "distance_to_next_m"}.
        distance_to_next_m is None if there's no active route or no GPS fix."""
        g = self.live_guidance()
        return {
            "step_index": g["step_index"],
            "total_steps": g["total_steps"],
            "next_text": g["text"],
            "distance_to_next_m": g["remaining_m"],
        }

    def tick(self):
        """Call periodically (e.g. once per second) from the main loop.

        Updates along-track progress, detects arrival, and reroutes only
        when the user is actually off the polyline (point-to-segment).
        Does NOT push ordinary turn instructions -- CombinedGuidanceLoop
        speaks those, with live remaining distance, as part of the combined
        phrase. MAP_CRITICAL is still pushed immediately for a true reroute
        so the user hears "đi lệch đường" without waiting for the next
        combined-cycle speech.
        """
        with self._lock:
            if not self.instructions or self._next_idx >= len(self.instructions):
                return
            geometry = list(self.route_geometry)
            cum = list(self._cum_dist)
            destination = self._destination
            instructions = list(self.instructions)

        loc = self.get_location_fn()
        if loc is None:
            return

        along, dist_path, _ = (None, None, None)
        if geometry:
            along, dist_path, _ = self._project_onto_route(loc, geometry, cum)

        # Advance _next_idx past maneuvers we have walked through.
        with self._lock:
            while self._next_idx < len(self.instructions):
                inst = self.instructions[self._next_idx]
                if inst.get("maneuver_type") == "arrive":
                    break
                a = inst.get("along_m")
                if along is not None and a is not None and along >= a + self.PASS_MANEUVER_M:
                    self._next_idx += 1
                    self._announced_current = False
                else:
                    break

        # Arrival: close to the arrive maneuver along-track (or haversine fallback).
        arrive_inst = None
        for inst in instructions:
            if inst.get("maneuver_type") == "arrive":
                arrive_inst = inst
                break
        if arrive_inst is not None:
            if along is not None and arrive_inst.get("along_m") is not None:
                remaining_arrive = abs(arrive_inst["along_m"] - along)
            elif arrive_inst.get("location") is not None:
                remaining_arrive = self._haversine_m(loc, tuple(arrive_inst["location"]))
            else:
                remaining_arrive = None
            if remaining_arrive is not None and remaining_arrive < self.ARRIVE_RADIUS_M:
                with self._lock:
                    self._next_idx = len(self.instructions)
                self.arrived_event.set()
                return

        # Off-route check: distance to the PATH (point-to-segment), not to
        # the next maneuver and not to the nearest vertex.
        if destination is not None and geometry and dist_path is not None and dist_path > self.OFF_ROUTE_M:
            self.audio_system.push(Priority.MAP_CRITICAL, "Bạn đã đi lệch đường, đang tìm lại đường đi.")
            self._route_to(loc, destination)


# ============================================================================
# 6b. SIMULATED GPS  (testing only, until a real phone GPS feed exists)
# ============================================================================

class SimulatedGpsWalker:
    """Stands in for a live GPS feed during testing: walks along the
    resolved route's geometry at a fixed pace so RouteGuide's progress /
    turn-announcement / arrival / reroute logic can all be exercised
    end-to-end without real hardware. RouteGuide only ever calls
    get_location_fn() -- it has no idea whether the fix is real or
    simulated, so swapping this for a real GPS callback later is a
    one-line change at the call site, not a RouteGuide change."""

    def __init__(self, speed_mps=1.2):
        self.speed_mps = speed_mps
        self._points = []
        self._cum_dist = [0.0]
        self._traveled_m = 0.0
        self._lock = threading.Lock()

    def set_route_geometry(self, coordinates, snap_lonlat=None):
        """Load a new polyline. If `snap_lonlat` is given, start from the
        nearest point on that polyline (so a reroute does not teleport the
        walker back to vertex 0 of a stale path)."""
        with self._lock:
            self._points = list(coordinates)
            self._cum_dist = RouteGuide._polyline_cum_dist(self._points) if self._points else [0.0]
            if snap_lonlat is not None and len(self._points) >= 2:
                along, _, _ = RouteGuide._project_onto_route(
                    snap_lonlat, self._points, self._cum_dist)
                self._traveled_m = 0.0 if along is None else along
            else:
                self._traveled_m = 0.0

    def advance(self, dt_s):
        with self._lock:
            if len(self._points) < 2:
                return
            self._traveled_m = min(self._traveled_m + self.speed_mps * dt_s, self._cum_dist[-1])

    def current_location(self):
        with self._lock:
            if not self._points:
                return None
            if len(self._points) == 1:
                return self._points[0]
            target = self._traveled_m
            for i in range(1, len(self._cum_dist)):
                if self._cum_dist[i] >= target:
                    seg_len = self._cum_dist[i] - self._cum_dist[i - 1]
                    frac = 0.0 if seg_len <= 0 else (target - self._cum_dist[i - 1]) / seg_len
                    lon1, lat1 = self._points[i - 1]
                    lon2, lat2 = self._points[i]
                    return (lon1 + (lon2 - lon1) * frac, lat1 + (lat2 - lat1) * frac)
            return self._points[-1]


# ============================================================================
# 7. VOICE DESTINATION HANDLER  (Function 0 -> RouteGuide bridge)
# ============================================================================

class VoiceDestinationHandler:
    """Bridges Function0_voice_control's PhoWhisper ASR to RouteGuide: a
    raw destination audio clip in -> a resolved route + started guidance.
    Loads the ASR model once and reuses it for every destination query,
    same as Function0's --serve mode."""

    def __init__(self, route_guide):
        self.route_guide = route_guide
        fv0.MODELS = fv0._load_asr()

    def handle_audio_file(self, audio_path):
        text_vi = fv0.speech_to_text_vi(Path(audio_path))
        self.route_guide.set_destination_by_name(text_vi)
        return text_vi


# ============================================================================
# 7b. COMBINED GUIDANCE LOOP  (one phrase: live nav + hazard warnings)
# ============================================================================

class CombinedGuidanceLoop(threading.Thread):
    """Unified GPS + detection + speech cycle.

    Every `cycle_interval_s` seconds (default 10):
      1. Read updated GPS (via RouteGuide.live_guidance) so remaining
         distance shrinks as the user moves.
      2. Run object/hazard detection on the latest frame.
      3. Compose ONE spoken phrase: [Updated Nav] + [Hazard Warning(s)].
      4. Push it once. DANGER uses HAZARD_DANGER (interrupts playback).

    Progress ticks (`RouteGuide.tick`) still run at `poll_interval_s`
    (default 1 s) so arrival / true off-route are not delayed by the
    cycle_interval_s speech cadence.

    Detection is never blocked on Mapbox HTTP or on TTS synthesis -- the
    only work in the cycle is local inference + a non-blocking queue put.

    NOTE on hazard latency: hazard detection runs once per
    `cycle_interval_s`, tied to the same cadence as nav announcements (so
    with the default 10s cycle, a newly-appeared DANGER can wait up to 10s
    before being spoken). If sub-second reaction to genuine emergencies
    matters more than battery/CPU budget, run HazardWorker (section 3)
    alongside this loop instead of/in addition to it -- it pushes hazard
    messages the moment they're detected, independent of the nav cadence.
    """

    def __init__(self, capture_fn, audio_system, route_guide,
                 cycle_interval_s=10.0, poll_interval_s=1.0, max_cycles=None,
                 own_pipeline=True):
        super().__init__(daemon=True)
        self.capture_fn = capture_fn
        self.audio_system = audio_system
        self.route_guide = route_guide
        self.cycle_interval_s = cycle_interval_s
        self.poll_interval_s = poll_interval_s
        self.max_cycles = max_cycles
        self._stop_event = threading.Event()
        self._last_spoken = None
        self._cycles = 0
        self._last_speak_at = 0.0
        self._start_time = time.time()
        self._pipeline_ready = False
        if own_pipeline:
            # This loop calls detect_hazards() once per cycle_interval_s
            # (10s here) -- much slower than the ~1-3s cadence
            # pl's tracker/confirm_frames machinery assumes. With
            # confirm_frames left at its video-loop default (2) and
            # max_track_match_time_s=6.0s, a track created in one cycle
            # ALWAYS times out before the next cycle can re-see it (10s
            # gap > 6s window) -- it can never reach 2 sightings, so
            # confirmed never becomes True, and every real hazard (e.g. a
            # motorcycle the detector clearly sees) is silently filtered
            # out of `messages` every single cycle. Each cycle here
            # already IS one independent snapshot, not a continuous video
            # stream, so it should confirm on first sighting -- same fix
            # pl.run_image_to_audio already applies for its single-frame
            # case.
            cfg = _pl_config()
            cfg["confirm_frames"] = 1
            # All messages built within one process_frame() call share the
            # same timestamp, so the default 0.4s min-gap-between-messages
            # throttle (meant to pace a live continuous stream) would
            # silently drop every message after the first one in that same
            # call. Zero it out, again matching run_image_to_audio.
            cfg["min_gap_between_any_announcement_s"] = 0.0
            (self.detector, self.depth_estimator, self.ground_plane, self.tracker,
             self.announcer, self.path_advisor) = pl.build_pipeline()
            self.stair_hole_classifier = pl.build_stair_hole_classifier()
            self._pipeline_ready = True
        else:
            self.detector = self.depth_estimator = self.ground_plane = None
            self.tracker = self.announcer = self.path_advisor = None
            self.stair_hole_classifier = None

    def stop(self):
        self._stop_event.set()

    def detect_hazards(self):
        """Run one Function 2 frame.

        Returns (messages, path_status_text):
          - messages: deduped list of announcement dicts (each with at
            least `text` and `warning`) -- one per (label, side, warning)
            group, nearest sighting kept, matching what Function2's own
            single-shot path already does before it speaks (see
            _dedupe_hazard_messages). Without this, the same tracked
            object can appear several times in one combined phrase.
          - path_status_text: best-effort Vietnamese phrase for the
            walkable-path status (e.g. "Đường bằng phẳng." when clear),
            or None if it can't be determined (see _format_path_status_vi).
        Both are ([], None) if there's no frame / no pipeline.
        """
        if not self._pipeline_ready:
            return [], None
        frame = self.capture_fn()
        if frame is None:
            return [], None
        timestamp = time.time()
        _, _, _, messages, path_status, _ = pl.process_frame(
            frame, self.detector, self.depth_estimator, self.ground_plane,
            self.tracker, self.announcer, timestamp, self.path_advisor,
            self.stair_hole_classifier,
        )
        messages = _dedupe_hazard_messages(messages or [])
        path_status_text = _format_path_status_vi(path_status)
        return messages, path_status_text

    def run_one_cycle(self, hazard_messages=None):
        """Sample GPS, detect (unless pre-supplied), compose, and push one
        merged phrase -- printing a diagnostic line for each stage so a
        cycle's full reasoning is visible in the console, e.g.:

            [CombinedLoop] Cycle 3 (t=20s)
            [GPS] lat=10.777100, lon=106.701500 | segment_dist_to_route=1.8m (ON ROUTE)
            [Detector] label='hole' conf=0.920 pos='AHEAD' -> HAZARD DETECTED (DANGER)
            [NavEngine] active_step='Đi 400 mét rồi rẽ phải.' | remaining_step_dist=240m
            [AudioBuilder] Merging output: 'Đi 240 mét rồi rẽ phải. Nguy hiểm, phía trước có hố sâu.'

        Returns the spoken text, or None if there was nothing new to say.
        """
        self._cycles += 1
        elapsed_s = time.time() - self._start_time
        print(f"\n[CombinedLoop] Cycle {self._cycles} (t={elapsed_s:.0f}s)")

        loc = self.route_guide.get_location_fn()
        dist_to_route, on_route = self.route_guide.distance_to_route(loc)
        if loc is not None:
            lon, lat = loc
            dist_txt = f"{dist_to_route:.1f}m" if dist_to_route is not None else "n/a"
            route_txt = "ON ROUTE" if on_route else ("OFF ROUTE" if on_route is not None else "NO ROUTE YET")
            print(f"[GPS] lat={lat:.6f}, lon={lon:.6f} | segment_dist_to_route={dist_txt} ({route_txt})")
        else:
            print("[GPS] no fix available")

        guidance = self.route_guide.live_guidance()
        nav_text = guidance.get("text")

        path_status_text = None
        if hazard_messages is None:
            hazard_messages, path_status_text = self.detect_hazards()

        if hazard_messages:
            for m in hazard_messages:
                label = m.get("class_name") or m.get("raw_label") or m.get("key", "hazard")
                conf = m.get("confidence")
                conf_txt = f"{conf:.3f}" if conf is not None else "n/a"
                pos = m.get("broad_side", "-")
                print(f"[Detector] label={label!r} conf={conf_txt} pos={pos!r} -> "
                      f"HAZARD DETECTED ({m.get('warning')})")
        else:
            print("[Detector] no hazard this cycle")

        print(f"[PathStatus] {path_status_text!r}" if path_status_text
              else "[PathStatus] not available this cycle")

        if nav_text:
            remaining = guidance.get("remaining_m")
            remaining_txt = f"{remaining:.0f}m" if remaining is not None else "n/a"
            print(f"[NavEngine] active_step={nav_text!r} | remaining_step_dist={remaining_txt}")
        else:
            print("[NavEngine] no active route step")

        text = compose_combined_utterance(hazard_messages, nav_text, path_status_text)
        if not text:
            print("[AudioBuilder] nothing to say this cycle")
            return None

        levels = [m.get("warning") for m in hazard_messages]
        if text == self._last_spoken and "DANGER" not in levels:
            print("[AudioBuilder] unchanged since last cycle -- not re-speaking")
            return None

        print(f"[AudioBuilder] Merging output: {text!r}")

        if "DANGER" in levels:
            priority = Priority.HAZARD_DANGER
        elif hazard_messages:
            priority = Priority.HAZARD_WARNING
        elif guidance.get("maneuver_type") == "arrive":
            priority = Priority.MAP_CRITICAL
        else:
            priority = Priority.MAP_STEP

        self.audio_system.push(priority, text, meta={
            "combined": True,
            "hazards": hazard_messages,
            "nav": guidance,
        })
        self._last_spoken = text
        return text

    def run(self):
        # First combined phrase fires immediately once a route exists; after
        # that we wait a full cycle_interval_s. Tick every poll_interval_s
        # so along-track / arrival / true-reroute stay responsive.
        self._last_speak_at = 0.0
        while not self._stop_event.is_set():
            self.route_guide.tick()
            now = time.time()
            if now - self._last_speak_at >= self.cycle_interval_s:
                self.run_one_cycle()
                self._last_speak_at = now
                if self.max_cycles is not None and self._cycles >= self.max_cycles:
                    break
            self._stop_event.wait(self.poll_interval_s)


# ============================================================================
# 8. ORCHESTRATOR
# ============================================================================

def run_navigation_session(capture_fn, get_location_fn, bluetooth_audio_fn,
                            mapbox_client, piper_voice_path=None, poll_interval_s=1.0,
                            detect_interval_s=None, cycle_interval_s=10.0,
                            audio_out_dir=None):
    """
    Wires everything together:
      - CombinedGuidanceLoop  GPS remaining-distance + Function 2 detection
                              + one spoken phrase every `cycle_interval_s`.
      - NavigationAudioSystem one priority queue + player thread; every
                              spoken message is saved as a real .wav under
                              `audio_out_dir` (if given) in addition to any
                              live playback.
      - RouteGuide            along-track progress, true off-route reroute.
      - VoiceDestinationHandler  new destination audio -> transcribe -> route.

    `detect_interval_s` is accepted as a deprecated alias for
    `cycle_interval_s` so existing call sites keep working.

    capture_fn()         -> next BGR frame (np.ndarray) or None
    get_location_fn()    -> (lon, lat) or None
    bluetooth_audio_fn() -> path to a new destination clip, or None
    """
    piper_voice_path = piper_voice_path or _pl_config()["piper_voice_path"]
    if detect_interval_s is not None:
        cycle_interval_s = detect_interval_s

    audio_system = NavigationAudioSystem(piper_voice_path, audio_out_dir=audio_out_dir)
    audio_system.start()

    route_guide = RouteGuide(mapbox_client, audio_system, get_location_fn)
    voice_handler = VoiceDestinationHandler(route_guide)

    loop = CombinedGuidanceLoop(
        capture_fn, audio_system, route_guide,
        cycle_interval_s=cycle_interval_s,
        poll_interval_s=poll_interval_s,
    )
    loop.start()

    try:
        while True:
            audio_path = bluetooth_audio_fn()
            if audio_path:
                voice_handler.handle_audio_file(audio_path)
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        loop.join(timeout=10.0)
        audio_system.stop()


# ============================================================================
# 9. LIVE TEST MODE  (continuous, runs until arrival or Ctrl+C)
# ============================================================================
# Default entry point: 1 photo + 1 destination-audio clip + 1 GPS fix in,
# but the SESSION itself runs continuously, exactly like the real device
# will --
#   - CombinedGuidanceLoop fires every `cycle_interval_s` (default 10 s):
#     sample GPS remaining distance, detect hazards on the test photo,
#     speak ONE combined phrase. Function 2 announcements are still
#     per-track-cooldown-limited, so an unchanging scene does not spam.
#   - The destination is transcribed and routed once, same as a single
#     spoken "take me to X" command on the device.
#   - SimulatedGpsWalker walks along the resolved route at a configurable
#     pace so remaining distance actually counts down.
#   - The session ends when the destination is reached (RouteGuide sets
#     arrived_event) OR you press Ctrl+C. Background threads are joined
#     before exit so a native-library call that's mid-flight finishes.
#
# mode="direct" is used here (a public/restricted Mapbox token passed on
# the command line or via the MAPBOX_TOKEN env var) since there's no
# backend yet to hold a secret token -- see MapboxClient's docstring for
# why that's dev-only and shouldn't ship this way.
# ============================================================================

def run_live_test(image_path, audio_path, origin_lonlat, mapbox_token,
                   piper_voice_path=None, walking_speed_mps=1.2,
                   tick_interval_s=1.0, detect_interval_s=None,
                   cycle_interval_s=10.0, out_dir="test_output"):
    piper_voice_path = piper_voice_path or _pl_config()["piper_voice_path"]
    if detect_interval_s is not None:
        cycle_interval_s = detect_interval_s
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(exist_ok=True, parents=True)
    audio_out_dir = out_dir_path / "audio"

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    def capture_fn():
        # stand-in for a live camera feed -- same static frame every cycle
        # until a real capture_fn is wired in
        return frame

    print("Starting live test session -- Ctrl+C stops it (stand-in for a physical stop button).")
    print(f"Spoken audio will be saved as real .wav files under: {audio_out_dir}")
    audio_system = NavigationAudioSystem(piper_voice_path, audio_out_dir=audio_out_dir)
    audio_system.start()

    print("\nTranscribing destination audio (Function 0)...")
    fv0.MODELS = fv0._load_asr()
    destination_text_vi = fv0.speech_to_text_vi(Path(audio_path))
    print(f"[Destination (vi)]: {destination_text_vi}")

    walker = SimulatedGpsWalker(speed_mps=walking_speed_mps)
    location_box = {"loc": origin_lonlat}

    def get_location_fn():
        return location_box["loc"]

    def _sync_walker_to_new_route(geometry):
        # Called by RouteGuide every time it resolves a route -- the FIRST
        # one and every REROUTE. Snapping to the current GPS (instead of
        # resetting traveled_m to 0) is what stops a reroute from
        # teleporting the walker back to the origin of the new polyline
        # when that origin is not where we already are.
        if not geometry:
            print("[SimulatedGpsWalker] new route had no usable geometry -- GPS will "
                  "stay fixed, so turn announcements won't advance.")
            return
        here = location_box["loc"]
        walker.set_route_geometry(geometry, snap_lonlat=here)
        location_box["loc"] = walker.current_location() or here
        print(f"[SimulatedGpsWalker] synced to route ({len(geometry)} geometry points).")

    mapbox = MapboxClient(mode="direct", public_token=mapbox_token)
    route_guide = RouteGuide(mapbox, audio_system, get_location_fn,
                              on_route_resolved=_sync_walker_to_new_route)

    print("\nResolving destination + fetching walking route (Mapbox)...")
    if not route_guide.set_destination_by_name(destination_text_vi):
        print("Could not resolve a route -- stopping.")
        audio_system.stop()
        return

    combined = CombinedGuidanceLoop(
        capture_fn, audio_system, route_guide,
        cycle_interval_s=cycle_interval_s,
        poll_interval_s=tick_interval_s,
        own_pipeline=True,
    )
    print(f"[CombinedGuidanceLoop] GPS + hazard + one spoken phrase every "
          f"{cycle_interval_s:.1f}s.")
    print(f"\nWalking simulation running at {walking_speed_mps} m/s, "
          f"ticking every {tick_interval_s}s. Listening for combined audio...\n")

    # Speak once immediately so the first remaining-distance + hazard phrase
    # is not delayed a full cycle after "Đã tìm thấy đường đi".
    combined.run_one_cycle()
    last_speak_at = time.time()

    tick_count = 0
    try:
        while not route_guide.arrived_event.is_set():
            walker.advance(tick_interval_s)
            location_box["loc"] = walker.current_location() or location_box["loc"]
            route_guide.tick()
            tick_count += 1
            now = time.time()
            if now - last_speak_at >= cycle_interval_s:
                combined.run_one_cycle()
                last_speak_at = now
            if tick_count % 5 == 0:
                # Periodic proof the simulation is actually moving: distance
                # to the next maneuver should be shrinking over time.
                p = route_guide.progress()
                if p["next_text"]:
                    dist_txt = f"{p['distance_to_next_m']:.0f}m" if p["distance_to_next_m"] is not None else "?"
                    print(f"[sim] step {p['step_index'] + 1}/{p['total_steps']} -- "
                          f"{dist_txt} to: {p['next_text']}")
            time.sleep(tick_interval_s)
        # Arrival cycle: make sure "Bạn đã đến nơi" is spoken, possibly
        # still combined with a last hazard.
        combined.run_one_cycle()
        print("\nDestination reached -- stopping.")
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C). Waiting for background threads to finish cleanly "
              "(press Ctrl+C again to force-exit immediately, skipping cleanup)...")
    finally:
        try:
            combined.stop()
            audio_system.stop()
            print(f"All spoken audio was saved under: {audio_out_dir}")
        except KeyboardInterrupt:
            print("\n[run_live_test] second Ctrl+C -- exiting immediately without waiting "
                  "for threads to finish (audio files synthesized so far are still on disk).")


# ============================================================================
# 10. SINGLE-SHOT TEST MODE  (quick smoke test -- --once)
# ============================================================================
# Runs each stage exactly once with no threads/queues/loop, and saves the
# hazard, map, AND combined audio as .wav files instead of speaking them
# live. Useful for a fast sanity check (e.g. "did Mapbox resolve this
# destination at all") without waiting through a full simulated walk.
# ============================================================================

def run_single_shot_test(image_path, audio_path, origin_lonlat, mapbox_token,
                          out_dir="test_output", piper_voice_path=None):
    piper_voice_path = piper_voice_path or _pl_config()["piper_voice_path"]
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("STEP 1/3 -- Function 2: hazard detection on the photo")
    print("=" * 70)
    hazard_messages, _path_status, hazard_wav, hazard_text = pl.run_image_to_audio(
        image_path,
        out_wav_path=str(out_dir / "hazard_audio.wav"),
        out_image_path=str(out_dir / "hazard_annotated.jpg"),
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("STEP 2/3 -- Function 0: transcribe destination audio")
    print("=" * 70)
    fv0.MODELS = fv0._load_asr()
    destination_text_vi = fv0.speech_to_text_vi(Path(audio_path))
    print(f"[Destination (vi)]: {destination_text_vi}")

    print("\n" + "=" * 70)
    print("STEP 3/3 -- Mapbox: resolve destination + walking directions")
    print("=" * 70)
    mapbox = MapboxClient(mode="direct", public_token=mapbox_token)
    first_nav = None
    try:
        hits = mapbox.search_box_forward(destination_text_vi, proximity_lonlat=origin_lonlat)
        if not hits:
            map_text = f"Không tìm thấy {destination_text_vi}."
        else:
            dest = hits[0]
            print(f"[Resolved place]: {dest['name']} ({dest['full_address']}) "
                  f"at {dest['longitude']},{dest['latitude']}")
            route = mapbox.directions_walking(origin_lonlat, (dest["longitude"], dest["latitude"]))
            instructions = build_step_instructions_vi(route)
            for i, step in enumerate(instructions, 1):
                print(f"  {i}. {step['text']}")
            map_text = " ".join(step["text"] for step in instructions)
            # Live-style first instruction: skip depart, use remaining of
            # the first real turn (the static step distance, since we are
            # still at the origin in --once mode).
            for step in instructions:
                if step.get("maneuver_type") not in SKIP_AS_NEXT_MANEUVER:
                    first_nav = live_step_text_vi(
                        step.get("maneuver_type"),
                        step.get("modifier"),
                        step.get("distance_m") or 0.0,
                    )
                    break
            if first_nav is None and instructions:
                first_nav = instructions[0]["text"]
    except Exception as e:
        print(f"[Mapbox error]: {e!r}")
        map_text = "Không tìm được đường đi, vui lòng thử lại."

    map_wav_path = str(out_dir / "map_audio.wav")
    tts = pl.PiperTTSVi(piper_voice_path)
    tts.speak_to_wav(map_text, map_wav_path)

    combined_text = compose_combined_utterance(hazard_messages, first_nav or map_text)
    combined_wav_path = str(out_dir / "combined_audio.wav")
    if combined_text:
        tts.speak_to_wav(combined_text, combined_wav_path)
        print(f"[audio] {combined_text}")
        print(f"[audio] saved: {combined_wav_path}")

    print("\n" + "=" * 70)
    print("SUMMARY  (live, nav + hazard are now ONE phrase; here they are")
    print("          also saved separately so you can listen to each)")
    print("=" * 70)
    print(f"Hazard audio : {hazard_wav}\n  text: {hazard_text}")
    print(f"Map audio    : {map_wav_path}\n  text: {map_text}")
    print(f"Combined     : {combined_wav_path}\n  text: {combined_text}")

    return {
        "hazard_messages": hazard_messages, "hazard_text": hazard_text, "hazard_wav": hazard_wav,
        "destination_text": destination_text_vi, "map_text": map_text, "map_wav": map_wav_path,
        "combined_text": combined_text, "combined_wav": combined_wav_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Function 3 test: 1 photo + 1 destination-audio clip + GPS -> combined nav + hazard guidance"
    )
    parser.add_argument("--image", required=True, help="Path to one test photo")
    parser.add_argument("--audio", required=True, help="Path to one spoken-destination audio clip")
    parser.add_argument("--lat", type=float, default=10.850555, help="Phone GPS latitude (default: yours)")
    parser.add_argument("--lon", type=float, default=106.81132027777778, help="Phone GPS longitude (default: yours)")
    parser.add_argument("--mapbox-token", default=os.environ.get("MAPBOX_TOKEN"),
                         help="Dev-only public/restricted Mapbox token (or set MAPBOX_TOKEN env var)")
    parser.add_argument("--out-dir", default="test_output",
                         help="Where to write .wav/.jpg outputs, including saved spoken audio under <out-dir>/audio")
    parser.add_argument("--walk-speed", type=float, default=1.2, help="Simulated walking speed, m/s")
    parser.add_argument("--tick-interval", type=float, default=1.0, help="Seconds between GPS/route ticks")
    parser.add_argument("--cycle-interval", type=float, default=10.0,
                         help="Seconds between combined nav+hazard spoken phrases (live test mode)")
    parser.add_argument("--detect-interval", type=float, default=None,
                         help="Deprecated alias for --cycle-interval")
    parser.add_argument("--once", action="store_true",
                         help="Quick smoke test: run each stage once, save .wav files, no live loop")
    args = parser.parse_args()

    if not args.mapbox_token:
        parser.error("Provide --mapbox-token or set the MAPBOX_TOKEN env var (a public/restricted "
                      "token is fine for this local test -- never a secret token)")

    origin_lonlat = (args.lon, args.lat)   # Mapbox wants (lon, lat), not (lat, lon)
    cycle_interval_s = args.detect_interval if args.detect_interval is not None else args.cycle_interval

    if args.once:
        run_single_shot_test(args.image, args.audio, origin_lonlat, args.mapbox_token, out_dir=args.out_dir)
    else:
        run_live_test(args.image, args.audio, origin_lonlat, args.mapbox_token,
                       walking_speed_mps=args.walk_speed, tick_interval_s=args.tick_interval,
                       cycle_interval_s=cycle_interval_s, out_dir=args.out_dir)

# ============================================================================
# 11. FASTAPI BACKEND SESSION INTEGRATION
# ============================================================================

class BackendNavigationSession:
    '''Stateful wrapper for the backend to run Function 3 map logic.'''
    def __init__(self, mapbox_token):
        import cv2
        self.mapbox = MapboxClient(mode='direct', public_token=mapbox_token)
        
        class MockAudioSystem:
            def __init__(self):
                self.messages = []
            def start(self): pass
            def stop(self): pass
            def push(self, priority, text, meta=None):
                self.messages.append(text)
                
        self.audio_system = MockAudioSystem()
        self.current_lat = None
        self.current_lon = None
        self.current_image_path = None
        
        def get_location_fn():
            if self.current_lat is not None and self.current_lon is not None:
                return (self.current_lon, self.current_lat)
            return None
            
        def capture_fn():
            if self.current_image_path:
                return cv2.imread(str(self.current_image_path))
            return None
            
        self.route_guide = RouteGuide(self.mapbox, self.audio_system, get_location_fn)
        
        self.combined_loop = CombinedGuidanceLoop(
            capture_fn, self.audio_system, self.route_guide,
            own_pipeline=True
        )

    def process_step(self, image_path, destination_text=None, lat=None, lon=None):
        self.current_image_path = image_path
        if lat is not None: self.current_lat = float(lat)
        if lon is not None: self.current_lon = float(lon)
        self.audio_system.messages.clear()
        
        if destination_text:
            self.route_guide.set_destination_by_name(destination_text)
            
        self.route_guide.tick()
        text = self.combined_loop.run_one_cycle()
        
        msgs = self.audio_system.messages
        if msgs:
            return " ".join(msgs)
        return text or "Đường đi phía trước thoáng."

ACTIVE_SESSIONS = {}

def process_navigation_step(session_id, mapbox_token, image_path, destination_text=None, lat=None, lon=None):
    if session_id not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[session_id] = BackendNavigationSession(mapbox_token)
    session = ACTIVE_SESSIONS[session_id]
    return session.process_step(image_path, destination_text, lat, lon)
