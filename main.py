"""
BlindAssist — Object Detection + Face Recognition + Emotion Voice Assistant
============================================================================
Reads directly from your webcam via OpenCV, detects objects with YOLO,
recognises known people by face (and whether they are smiling / sad /
neutral), and speaks using offline TTS (pyttsx3).

SETUP:
    pip install -U ultralytics opencv-python pyttsx3 hsemotion-onnx
    (OpenCV 4.8 or newer is required for the face models)

    # Windows: pyttsx3 uses SAPI5, works out of the box.
    # Linux:   sudo apt install espeak

KNOWN FACES (put your photos here, next to main.py):
    known_faces/
        Nishchay/
            img1.jpg
            img2.jpg
            ... (your 17 photos)
        MD.Eliyas Sheikh/
            img1.jpg
            img2.jpg
            ...

    The FOLDER NAME is the name that will be spoken. To add another person,
    just add another folder (e.g. known_faces/Rahul/) with their photos.

MODELS (downloaded automatically on first run, internet needed once):
    * Face detector + face recogniser (OpenCV ONNX)  -> saved in ./models/
    * Emotion model (HSEmotion)                      -> handled by hsemotion-onnx
    If a download fails, the program tells you what to do.

WHAT IT SAYS (examples):
    "Nishchay is detected on your left and he is smiling."
    "Nishchay is now ahead of you and he is neutral."
    "Nishchay is now sad."
    "Nishchay is gone."

RUN:
    python main.py

Press 'q' in the video window (or Ctrl+C in the terminal) to stop.
"""

import logging
import shutil
import threading
import time
import traceback
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pyttsx3
import torch
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CUSTOM OBJECT CATEGORIES
# ============================================================

DAILY_OBJECTS = {
    "pen", "pencil", "book", "notebook", "remote", "keys", "wallet",
    "earphones", "charger", "mug", "glasses", "umbrella",
}

IMPORTANT_OBJECTS = {
    "medicine_box",
}

DANGEROUS_OBJECTS = {
    "knife", "scissors", "blade", "cutter", "needle", "syringe",
    "broken_glass", "hammer",
}

Box = Tuple[int, int, int, int, str, float, Optional[float], str]


@dataclass(frozen=True)
class Config:
    model_path: str = "yolov8n.pt"
    confidence: float = 0.5
    inference_size: int = 640
    detection_interval: float = 0.5       # seconds between detection passes
    camera_index: int = 0
    frame_width: int = 680
    frame_height: int = 480
    warmup_frames: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- announcement behaviour (objects) ---
    arrival_confirm_passes: int = 1       # passes an object must be seen before "detected"
    gone_after_seconds: float = 2.0       # not seen for this long -> "gone"
    direction_confirm_passes: int = 2     # passes in a new direction before "now left/right"
    direction_cooldown: float = 3.0       # min seconds between direction announcements
    max_pending_speech: int = 4           # speech backlog cap (old, non-urgent dropped)

    # --- face recognition ---
    faces_dir: str = "known_faces"        # relative to main.py (or absolute path)
    models_dir: str = "models"            # where the face ONNX models are stored
    default_person: str = "MD.Eliyas Sheikh"  # folder auto-created on first run
    face_det_score: float = 0.7           # YuNet face detector confidence
    face_match_threshold: float = 0.36    # cosine similarity needed to accept a match
    face_top_k: int = 3                   # average the best K template scores
    min_face_px: int = 40                 # ignore faces smaller than this (too unreliable)
    known_confirm_passes: int = 2         # passes a known face must be confirmed before announcing
    known_gone_after_seconds: float = 3.0 # not seen for this long -> "<name> is gone"
    identity_hold_seconds: float = 5.0    # keep tracking him by body after the face is lost
    person_arrival_confirm_passes: int = 2  # generic "person" waits so a face can be matched first
    face_debug: bool = False              # print match scores for unmatched faces (for tuning)

    # --- emotion recognition (known people only) ---
    emotion_enabled: bool = True
    emotion_model: str = "enet_b0_8_best_vgaf"
    emotion_window: int = 5               # last N readings are kept for smoothing
    emotion_min_votes: int = 3            # readings (out of the window) needed to switch emotion
    emotion_cooldown: float = 4.0         # min seconds between emotion-change announcements
    announce_emotion_changes: bool = True # say "Nishchay is now sad." when the emotion changes


CFG = Config()


# ============================================================
# SPEECH
# ============================================================

class SpeechEngine:
    """
    Single worker thread that speaks queued messages.

      * A FRESH pyttsx3 engine is created for every utterance. Re-using one
        engine inside a thread makes pyttsx3 go silent after a few calls
        (very common on Windows/SAPI5).
      * Backlog is capped and de-duplicated, so a pile of old messages can
        never delay new ones. Danger messages jump the queue.
    """

    def __init__(self, max_pending: int = 4) -> None:
        self._cond = threading.Condition()
        self._pending: Deque[Tuple[bool, str]] = deque()  # (urgent, text)
        self._max_pending = max_pending
        self._closed = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str, urgent: bool = False) -> None:
        with self._cond:
            if self._closed:
                return
            if any(t == text for _, t in self._pending):
                return  # identical message already waiting
            if urgent:
                self._pending.appendleft((True, text))
            else:
                self._pending.append((False, text))

            while len(self._pending) > self._max_pending:
                for i, (is_urgent, _) in enumerate(self._pending):
                    if not is_urgent:
                        del self._pending[i]   # drop oldest non-urgent
                        break
                else:
                    self._pending.pop()        # all urgent: drop the oldest
            self._cond.notify()

    def _worker(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                _, text = self._pending.popleft()
            self._say(text)

    @staticmethod
    def _say(text: str) -> None:
        for attempt in range(2):
            engine = None
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                return
            except Exception as e:
                print(f"[TTS ERROR] attempt {attempt + 1}: {e}")
            finally:
                if engine is not None:
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    del engine

    def stop(self) -> None:
        with self._cond:
            self._closed = True
            self._pending.clear()
            self._cond.notify_all()


# ============================================================
# FACE RECOGNITION  (OpenCV YuNet detector + SFace recogniser)
# ============================================================

DET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
REC_MODEL_NAME = "face_recognition_sface_2021dec.onnx"

MODEL_URLS: Dict[str, List[str]] = {
    DET_MODEL_NAME: [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + DET_MODEL_NAME,
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/" + DET_MODEL_NAME,
    ],
    REC_MODEL_NAME: [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/" + REC_MODEL_NAME,
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/" + REC_MODEL_NAME,
    ],
}

MIN_MODEL_BYTES = 100_000
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else BASE_DIR / path


def _ensure_model(path: Path) -> None:
    """Make sure an ONNX model exists locally; download it if missing."""
    if path.exists() and path.stat().st_size > MIN_MODEL_BYTES:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for url in MODEL_URLS[path.name]:
        tmp = path.with_suffix(".part")
        try:
            print(f"[FACE] Downloading {path.name} ...")
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            if tmp.stat().st_size < MIN_MODEL_BYTES:
                raise RuntimeError("downloaded file is too small (not the real model)")
            tmp.replace(path)
            return
        except Exception as e:  # try next URL
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    raise RuntimeError(
        f"Could not download {path.name} ({last_err}).\n"
        f"        Download it manually from:\n"
        f"        {MODEL_URLS[path.name][0]}\n"
        f"        and save it as: {path}"
    )


class FaceIdentifier:
    """Enrols faces from known_faces/<Name>/*.jpg and identifies them live."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.detector = None
        self.recognizer = None
        self.templates: Dict[str, np.ndarray] = {}   # name -> (N, 128) unit vectors
        try:
            self._load()
        except Exception as e:
            print(f"[FACE] Face recognition DISABLED: {e}")
            self.detector = None
            self.recognizer = None
            self.templates = {}

    @property
    def ready(self) -> bool:
        return (self.detector is not None
                and self.recognizer is not None
                and bool(self.templates))

    # ---------- loading / enrolment ----------

    def _load(self) -> None:
        if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
            raise RuntimeError(
                "your OpenCV is too old for face recognition. "
                "Run: pip install -U opencv-python"
            )

        models_dir = _resolve_path(self.cfg.models_dir)
        det_path = models_dir / DET_MODEL_NAME
        rec_path = models_dir / REC_MODEL_NAME
        _ensure_model(det_path)
        _ensure_model(rec_path)

        self.detector = cv2.FaceDetectorYN.create(
            str(det_path), "", (320, 320), self.cfg.face_det_score, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self._enroll()

    @staticmethod
    def _read_image(path: Path) -> Optional[np.ndarray]:
        # np.fromfile + imdecode also works for non-ASCII Windows paths.
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None
        if img is None:
            return None
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > 1280:
            scale = 1280.0 / longest
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    def _embed_face(self, img: np.ndarray, face: np.ndarray) -> Optional[np.ndarray]:
        aligned = self.recognizer.alignCrop(img, face)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm == 0.0:
            return None
        return feat / norm

    def _embed_largest_face(self, img: np.ndarray) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        face = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
        return self._embed_face(img, face)

    def _enroll(self) -> None:
        faces_dir = _resolve_path(self.cfg.faces_dir)
        faces_dir.mkdir(parents=True, exist_ok=True)

        person_dirs = sorted(p for p in faces_dir.iterdir() if p.is_dir())
        if not person_dirs:
            default_dir = faces_dir / self.cfg.default_person
            default_dir.mkdir(parents=True, exist_ok=True)
            person_dirs = [default_dir]

        stray = [p for p in faces_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if stray:
            print(f"[FACE] WARNING: {len(stray)} image(s) sit directly inside {faces_dir}. "
                  f"Move them into a person folder, e.g. {faces_dir / self.cfg.default_person}")

        for person_dir in person_dirs:
            name = person_dir.name.strip()
            if not name or name.lower() == "person":
                continue

            feats: List[np.ndarray] = []
            used = total = 0
            for path in sorted(person_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                    continue
                total += 1
                img = self._read_image(path)
                if img is None:
                    print(f"[FACE]   skip {name}/{path.name}: cannot read image")
                    continue
                try:
                    f = self._embed_largest_face(img)
                    if f is None:
                        print(f"[FACE]   skip {name}/{path.name}: no face found")
                        continue
                    feats.append(f)
                    used += 1
                    # Mirrored copy: helps recognise side angles from either side.
                    f_flip = self._embed_largest_face(cv2.flip(img, 1))
                    if f_flip is not None:
                        feats.append(f_flip)
                except Exception as e:
                    print(f"[FACE]   skip {name}/{path.name}: {e}")

            if feats:
                self.templates[name] = np.vstack(feats)
                print(f"[FACE] Enrolled '{name}': {used}/{total} images used "
                      f"({len(feats)} face templates)")
                if used < 5:
                    print(f"[FACE]   WARNING: only {used} usable images for '{name}'. "
                          f"More clear, well-lit photos give better recognition.")
            else:
                print(f"[FACE] No usable face images for '{name}'. "
                      f"Put clear photos of the person in: {person_dir}")

        if not self.templates:
            raise RuntimeError(
                f"no usable face images found. Add photos to {faces_dir / self.cfg.default_person}"
            )

    # ---------- live identification ----------

    def identify(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, Optional[str], float]]:
        """
        Returns [(x1, y1, x2, y2, name_or_None, score), ...], best score first.
        name is None when the face does not match anyone enrolled.
        """
        if not self.ready:
            return []

        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        results: List[Tuple[int, int, int, int, Optional[str], float]] = []
        for face in faces:
            fw, fh = float(face[2]), float(face[3])
            if fw < self.cfg.min_face_px or fh < self.cfg.min_face_px:
                continue
            try:
                feat = self._embed_face(frame, face)
            except cv2.error:
                continue
            if feat is None:
                continue

            best_name: Optional[str] = None
            best_score = -1.0
            for name, templates in self.templates.items():
                sims = templates @ feat
                k = min(self.cfg.face_top_k, len(sims))
                score = float(np.mean(np.sort(sims)[-k:]))
                if score > best_score:
                    best_name, best_score = name, score

            x1 = max(0, int(face[0]))
            y1 = max(0, int(face[1]))
            x2 = min(w - 1, int(face[0] + fw))
            y2 = min(h - 1, int(face[1] + fh))

            if best_score >= self.cfg.face_match_threshold:
                results.append((x1, y1, x2, y2, best_name, best_score))
            else:
                if self.cfg.face_debug:
                    print(f"[FACE] unmatched face: best={best_name} score={best_score:.2f} "
                          f"(threshold {self.cfg.face_match_threshold})")
                results.append((x1, y1, x2, y2, None, best_score))

        results.sort(key=lambda r: r[5], reverse=True)
        return results


# ============================================================
# EMOTION RECOGNITION  (HSEmotion ONNX, collapsed to 3 classes)
# ============================================================

# The model knows 8 emotions; we only report these three.
EMOTION_MAP = {
    "happiness": "smiling",
    "happy": "smiling",
    "sadness": "sad",
    "sad": "sad",
    "neutral": "neutral",
}


class EmotionRecognizer:
    """
    Classifies a face crop as 'smiling', 'sad' or 'neutral'.
    Other emotions (anger, fear, surprise, ...) are mapped to whichever of the
    three the model scores highest.
    """

    def __init__(self, config: Config, enabled: bool = True) -> None:
        self.cfg = config
        self.fer = None
        self._idx_to_class: Optional[dict] = None
        if not enabled:
            return

        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        except ImportError:
            print("[EMOTION] Emotion recognition DISABLED: package 'hsemotion-onnx' is not "
                  "installed. Run: pip install hsemotion-onnx")
            return

        try:
            fer = HSEmotionRecognizer(model_name=config.emotion_model)
            # Self-test at startup so any problem shows up now, not mid-run.
            fer.predict_emotions(np.zeros((224, 224, 3), dtype=np.uint8), logits=True)
            self.fer = fer
            idx_to_class = getattr(fer, "idx_to_class", None)
            self._idx_to_class = idx_to_class if isinstance(idx_to_class, dict) else None
            print(f"[EMOTION] Emotion recognition ready ({config.emotion_model}).")
        except Exception as e:
            print(f"[EMOTION] Emotion recognition DISABLED: {e}")
            self.fer = None

    @property
    def ready(self) -> bool:
        return self.fer is not None

    def predict(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[str]:
        """box = (x1, y1, x2, y2) of the face in the BGR frame."""
        if self.fer is None:
            return None

        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(fw, x2 + pad_x), min(fh, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        face_rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        label, scores = self.fer.predict_emotions(face_rgb, logits=True)
        return self._to_three_class(label, scores)

    def _to_three_class(self, label, scores) -> Optional[str]:
        # Preferred: compare the model's scores for happiness / sadness / neutral.
        if self._idx_to_class:
            vec = np.asarray(scores, dtype=np.float32).reshape(-1)
            best: Optional[Tuple[str, float]] = None
            for idx, class_name in self._idx_to_class.items():
                mapped = EMOTION_MAP.get(str(class_name).lower())
                if mapped is None or int(idx) >= len(vec):
                    continue
                value = float(vec[int(idx)])
                if best is None or value > best[1]:
                    best = (mapped, value)
            if best is not None:
                return best[0]

        # Fallback: use the model's own top label if it is one of the three.
        return EMOTION_MAP.get(str(label).lower())


# ============================================================
# CAMERA
# ============================================================

class BlindAssistCamera:
    """Wraps a local webcam via OpenCV."""

    def __init__(self, config: Config):
        self.cfg = config
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        opened = False
        for index in [self.cfg.camera_index, 1 - self.cfg.camera_index]:
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                opened = True
                break
            self.cap.release()

        if not opened:
            for index in [self.cfg.camera_index, 1 - self.cfg.camera_index]:
                self.cap = cv2.VideoCapture(index)
                if self.cap.isOpened():
                    opened = True
                    break
                self.cap.release()

        if not opened:
            raise RuntimeError(
                f"Could not open webcam at index {self.cfg.camera_index}. "
                "Check that it's connected and not in use by another app."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: requested {self.cfg.frame_width}x{self.cfg.frame_height}, "
              f"actual {actual_w}x{actual_h}")

        for _ in range(self.cfg.warmup_frames):
            self.cap.read()

    def is_running(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_running():
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


# ============================================================
# APP
# ============================================================

class BlindAssistApp:
    """Detect -> identify faces + emotion -> track presence -> announce."""

    def __init__(self, config: Config):
        self.cfg = config
        self.camera = BlindAssistCamera(config)
        self.speech = SpeechEngine(max_pending=config.max_pending_speech)
        print(f"Loading {config.model_path} on {config.device}...")
        self.model = YOLO(config.model_path)
        self.face_id = FaceIdentifier(config)
        self.emotion = EmotionRecognizer(
            config, enabled=(config.emotion_enabled and self.face_id.ready)
        )

        # label -> tracking state
        self.tracks: Dict[str, dict] = {}
        # known person name -> {"face_time", "box", "score"}
        self._known_state: Dict[str, dict] = {}

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_boxes: List[Box] = []
        self._frame_lock = threading.Lock()
        self._box_lock = threading.Lock()
        self._detector_stop = threading.Event()

    # ---------------- helpers ----------------

    @property
    def known_names(self) -> Set[str]:
        return set(self.face_id.templates.keys())

    @staticmethod
    def _speakable(label: str) -> str:
        return label.replace("_", " ")

    @staticmethod
    def _location_phrase(direction: str) -> str:
        return "ahead of you" if direction == "ahead" else f"on your {direction}"

    @staticmethod
    def _emotion_tail(emotion: Optional[str]) -> str:
        return f" and he is {emotion}" if emotion else ""

    @staticmethod
    def _most_common(readings) -> Optional[str]:
        if not readings:
            return None
        return Counter(readings).most_common(1)[0][0]

    def _emotion_candidate(self, readings) -> Optional[str]:
        """The emotion that holds at least emotion_min_votes of the recent readings."""
        if not readings:
            return None
        label, votes = Counter(readings).most_common(1)[0]
        return label if votes >= self.cfg.emotion_min_votes else None

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / union if union > 0 else 0.0

    def _get_direction(self, x1: int, x2: int, frame_width: int) -> str:
        object_center = (x1 + x2) // 2
        if object_center < frame_width * 0.33:
            return "left"
        elif object_center > frame_width * 0.66:
            return "right"
        return "ahead"

    def _estimate_distance(self, label: str, box_width: int) -> Optional[float]:
        REAL_WIDTHS = {
            "person": 0.45,
            "cell phone": 0.075,
            "bottle": 0.07,
            "laptop": 0.32,
            "chair": 0.45,
            "backpack": 0.30,
            "cup": 0.08,
        }
        if box_width <= 0:
            return None
        real_width = REAL_WIDTHS.get(label.lower())
        if real_width is None:
            return None
        FOCAL_LENGTH = 700
        return round((real_width * FOCAL_LENGTH) / box_width, 1)

    def _get_object_category(self, label: str) -> str:
        label = label.lower().strip()
        if label in DANGEROUS_OBJECTS:
            return "dangerous"
        if label in IMPORTANT_OBJECTS:
            return "important"
        if label in DAILY_OBJECTS:
            return "daily"
        return "normal"

    def _create_voice_message(self, label: str, direction: str) -> str:
        category = self._get_object_category(label)
        name = self._speakable(label)
        location = self._location_phrase(direction)
        if category == "dangerous":
            return f"Warning! {name} detected {location}. Please be careful."
        if category == "important":
            return f"{name} detected {location}. Please take note."
        return f"{name} detected {location}."

    def _confirm_passes(self, label: str) -> int:
        if label in self.known_names:
            return self.cfg.known_confirm_passes
        if label == "person" and self.face_id.ready:
            # Wait a little so a face can be matched before saying a generic "person".
            return self.cfg.person_arrival_confirm_passes
        return self.cfg.arrival_confirm_passes

    # ---------------- detection ----------------

    def _detect_objects(self, frame: np.ndarray) -> List[Box]:
        results = self.model(
            frame,
            conf=self.cfg.confidence,
            imgsz=self.cfg.inference_size,
            device=self.cfg.device,
            verbose=False,
        )

        names = self.model.names
        boxes: List[Box] = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = names[int(box.cls[0])]
                conf = float(box.conf[0])
                distance = self._estimate_distance(label, x2 - x1)
                direction = self._get_direction(x1, x2, frame.shape[1])
                boxes.append((x1, y1, x2, y2, label, conf, distance, direction))

        return boxes

    def _apply_identities(
        self, frame: np.ndarray, boxes: List[Box]
    ) -> Tuple[List[Box], bool, Dict[str, str]]:
        """
        Relabel the YOLO 'person' box that contains a recognised face with the
        person's name (e.g. 'Nishchay'). If the face is recognised but YOLO has
        no matching person box, a box around the face is added instead.

        Returns (new_boxes, person_absorbed, emotions).
          person_absorbed: True when a generic 'person' box was just turned into
            a newly identified known person, so the tracker can drop the generic
            'person' silently instead of saying 'person gone'.
          emotions: {name: 'smiling' | 'sad' | 'neutral'} read from this frame
            (only for people whose face was actually seen this pass).
        """
        if not self.face_id.ready:
            return boxes, False, {}

        now = time.time()
        width = frame.shape[1]
        out: List[Box] = list(boxes)
        claimed: Set[int] = set()
        seen: Set[str] = set()
        emotions: Dict[str, str] = {}
        absorbed = False

        for fx1, fy1, fx2, fy2, name, score in self.face_id.identify(frame):
            if name is None or name in seen:
                continue

            cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            idx: Optional[int] = None
            best_area = 0
            for i, b in enumerate(out):
                if b[4] != "person" or i in claimed:
                    continue
                if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                    area = (b[2] - b[0]) * (b[3] - b[1])
                    if idx is None or area < best_area:
                        idx, best_area = i, area

            if idx is not None:
                x1, y1, x2, y2 = out[idx][:4]
                distance = self._estimate_distance("person", x2 - x1)
                direction = self._get_direction(x1, x2, width)
                out[idx] = (x1, y1, x2, y2, name, score, distance, direction)
                claimed.add(idx)
                if not self.tracks.get(name, {}).get("announced"):
                    absorbed = True
            else:
                x1, y1, x2, y2 = fx1, fy1, fx2, fy2
                direction = self._get_direction(x1, x2, width)
                out.append((x1, y1, x2, y2, name, score, None, direction))

            seen.add(name)
            self._known_state[name] = {
                "face_time": now,
                "box": (x1, y1, x2, y2),
                "score": score,
            }

            # Emotion is read from the FACE box (not the body box).
            if self.emotion.ready:
                try:
                    reading = self.emotion.predict(frame, (fx1, fy1, fx2, fy2))
                except Exception as e:
                    print(f"[EMOTION ERROR] {e}")
                    reading = None
                if reading:
                    emotions[name] = reading

        # Face not visible this pass (turned away / blurred) but he was confirmed
        # moments ago: if exactly one unclaimed person box overlaps where he was,
        # keep treating that box as him. Limited to identity_hold_seconds so a
        # stranger can never inherit the name for long.
        for name, st in self._known_state.items():
            if name in seen:
                continue
            if now - st["face_time"] > self.cfg.identity_hold_seconds:
                continue
            cands = [i for i, b in enumerate(out) if b[4] == "person" and i not in claimed]
            if len(cands) != 1:
                continue
            i = cands[0]
            b = out[i]
            if self._iou(b[:4], st["box"]) < 0.3:
                continue
            x1, y1, x2, y2 = b[:4]
            out[i] = (x1, y1, x2, y2, name, st["score"],
                      self._estimate_distance("person", x2 - x1),
                      self._get_direction(x1, x2, width))
            claimed.add(i)
            st["box"] = (x1, y1, x2, y2)

        return out, absorbed, emotions

    # ---------------- announcements ----------------

    def _update_tracking(
        self,
        boxes: List[Box],
        person_absorbed: bool = False,
        emotions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Announce arrivals, direction/emotion changes and objects/people that are gone."""
        now = time.time()
        known = self.known_names
        emotions = emotions or {}

        # Best (highest-confidence) detection per label in this pass.
        best: Dict[str, Tuple[float, str]] = {}
        for _x1, _y1, _x2, _y2, label, conf, _dist, direction in boxes:
            if label not in best or conf > best[label][0]:
                best[label] = (conf, direction)

        # ---- arrivals + movement + emotion ----
        for label, (_conf, direction) in best.items():
            is_known = label in known
            is_danger = label in DANGEROUS_OBJECTS
            st = self.tracks.get(label)

            if st is None:
                st = {
                    "last_seen": now,
                    "direction": direction,
                    "hits": 1,
                    "announced": False,
                    "pending_dir": None,
                    "pending_count": 0,
                    "last_dir_say": now,
                    # emotion state (used for known people only)
                    "emo_hist": deque(maxlen=max(1, self.cfg.emotion_window)),
                    "emotion": None,
                    "last_emo_say": now,
                }
                self.tracks[label] = st
            else:
                st["last_seen"] = now
                st["hits"] += 1

            if is_known and emotions.get(label):
                st["emo_hist"].append(emotions[label])

            # ----- arrival -----
            if not st["announced"]:
                if st["hits"] >= self._confirm_passes(label):
                    st["announced"] = True
                    st["direction"] = direction
                    st["last_dir_say"] = now
                    if is_known:
                        st["emotion"] = self._most_common(st["emo_hist"])
                        st["last_emo_say"] = now
                        msg = (f"{label} is detected {self._location_phrase(direction)}"
                               f"{self._emotion_tail(st['emotion'])}.")
                        tag = "PERSON"
                    else:
                        msg = self._create_voice_message(label, direction)
                        tag = "DANGER" if is_danger else "NEW"
                    print(f"[{tag}] [SPEAK] {msg}")
                    self.speech.speak(msg, urgent=is_danger)
                continue

            # ----- emotion change (known people) -----
            emotion_changed = False
            if is_known:
                cand = self._emotion_candidate(st["emo_hist"])
                if cand is not None and cand != st["emotion"]:
                    if st["emotion"] is None or now - st["last_emo_say"] >= self.cfg.emotion_cooldown:
                        st["emotion"] = cand
                        st["last_emo_say"] = now
                        emotion_changed = True

            # ----- direction change -----
            said_direction = False
            if direction == st["direction"]:
                st["pending_dir"] = None
                st["pending_count"] = 0
            else:
                if st["pending_dir"] == direction:
                    st["pending_count"] += 1
                else:
                    st["pending_dir"] = direction
                    st["pending_count"] = 1

                if (st["pending_count"] >= self.cfg.direction_confirm_passes
                        and now - st["last_dir_say"] >= self.cfg.direction_cooldown):
                    st["direction"] = direction
                    st["pending_dir"] = None
                    st["pending_count"] = 0
                    st["last_dir_say"] = now
                    name = self._speakable(label)
                    if is_known:
                        msg = (f"{label} is now {self._location_phrase(direction)}"
                               f"{self._emotion_tail(st['emotion'])}.")
                    elif is_danger:
                        msg = f"Warning! {name} is now {direction}. Please be careful."
                    else:
                        msg = f"{name} is now {direction}."
                    print(f"[MOVE] [SPEAK] {msg}")
                    self.speech.speak(msg, urgent=is_danger)
                    said_direction = True

            # Announce the emotion change unless the direction message already
            # carried the new emotion.
            if (emotion_changed and not said_direction
                    and self.cfg.announce_emotion_changes and st["emotion"]):
                msg = f"{label} is now {st['emotion']}."
                print(f"[EMOTION] [SPEAK] {msg}")
                self.speech.speak(msg)

        # ---- gone ----
        for label in list(self.tracks.keys()):
            if label in best:
                continue
            st = self.tracks[label]

            if not st["announced"]:
                # Never announced (seen too briefly) -> just forget it silently.
                del self.tracks[label]
                continue

            if label == "person" and person_absorbed:
                # That generic 'person' was just identified by name: not a departure.
                del self.tracks[label]
                continue

            is_known = label in known
            gone_after = (self.cfg.known_gone_after_seconds if is_known
                          else self.cfg.gone_after_seconds)
            if now - st["last_seen"] >= gone_after:
                if is_known:
                    msg = f"{label} is gone."
                else:
                    msg = f"{self._speakable(label)} gone."
                print(f"[GONE] [SPEAK] {msg}")
                self.speech.speak(msg, urgent=label in DANGEROUS_OBJECTS)
                del self.tracks[label]

    # ---------------- drawing ----------------

    def _draw_boxes(self, frame: np.ndarray, boxes: List[Box]) -> np.ndarray:
        known = self.known_names
        for x1, y1, x2, y2, label, conf, distance, direction in boxes:
            is_known = label in known
            color = (0, 165, 255) if is_known else (0, 255, 0)   # orange for known people
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if distance is not None:
                text = f"{label} {conf:.2f} | ~{distance}m | {direction}"
            else:
                text = f"{label} {conf:.2f} | {direction}"

            if is_known:
                emo = self.tracks.get(label, {}).get("emotion")
                if emo:
                    text += f" | {emo}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_top = max(y1, th + 10)   # keep the label on-screen
            cv2.rectangle(frame, (x1, y_top - th - 8), (x1 + tw + 4, y_top), color, -1)
            cv2.putText(frame, text, (x1 + 2, y_top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame

    # ---------------- threads ----------------

    def _detection_worker(self) -> None:
        last_detection_time = 0.0
        while not self._detector_stop.is_set():
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            if time.time() - last_detection_time < self.cfg.detection_interval:
                time.sleep(0.01)
                continue

            # Never let an exception kill this thread (that would silence
            # all announcements and freeze the boxes).
            try:
                boxes = self._detect_objects(frame)
                absorbed = False
                emotions: Dict[str, str] = {}
                try:
                    boxes, absorbed, emotions = self._apply_identities(frame, boxes)
                except Exception:
                    print("[FACE ERROR] continuing with object detection only")
                    traceback.print_exc()
                self._update_tracking(boxes, absorbed, emotions)
                with self._box_lock:
                    self._latest_boxes = boxes
            except Exception:
                print("[DETECTOR ERROR]")
                traceback.print_exc()

            last_detection_time = time.time()

    def run(self) -> None:
        self.camera.start()
        print("BlindAssist running. Press 'q' in the video window to stop.")
        self.speech.speak("BlindAssist is ready.")

        if self.face_id.ready:
            print(f"Face recognition active for: {', '.join(sorted(self.known_names))}")
            if self.cfg.emotion_enabled and not self.emotion.ready:
                print("[EMOTION] Emotion recognition is NOT active (see messages above).")
                self.speech.speak("Emotion recognition is not available.")
        else:
            print("[FACE] Face recognition is NOT active (see messages above).")
            self.speech.speak("Face recognition is not available.")

        detector_thread = threading.Thread(target=self._detection_worker, daemon=True)
        detector_thread.start()

        try:
            while self.camera.is_running():
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                with self._frame_lock:
                    self._latest_frame = frame

                with self._box_lock:
                    boxes = list(self._latest_boxes)

                display_frame = self._draw_boxes(frame.copy(), boxes)
                cv2.imshow("BlindAssist", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            self._detector_stop.set()
            detector_thread.join(timeout=2.0)
            self.speech.stop()
            self.camera.stop()
            print("BlindAssist stopped.")


if __name__ == "__main__":
    app = BlindAssistApp(CFG)
    app.run()