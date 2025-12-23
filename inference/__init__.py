"""Inference pipeline with optional local webcam capture and a simple detector.

This module provides a development-friendly `InferencePipeline` that will:
- Open a local webcam (if `video_reference` is an integer) and read frames
- Run a small, dependency-light detector that looks for a circular ball (via
  HoughCircles) and moving contours as player candidates
- Call `on_prediction(result, video_frame)` with a `result` dict containing
  `output_image` (with `numpy_image`) and `detections` (list of dicts)

If `cv2` or `numpy` are not available the pipeline falls back to a synthetic
frame generator (same behavior as the previous shim).
"""

from __future__ import annotations

import threading
import time
import traceback
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - environment may not have numpy
    np = None  # type: ignore

try:
    import cv2 as _cv2
except Exception:
    _cv2 = None  # type: ignore


class OutputImage:
    """Simple container to match expected interface: has attribute `numpy_image`."""

    def __init__(self, arr: Any):
        self.numpy_image = arr


Detection = Dict[str, Any]


class InferencePipeline:
    """Pipeline that can read from webcam or produce synthetic frames and detect objects."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        video_reference: Any = 0,
        max_fps: int = 30,
        on_prediction: Optional[Callable[[dict, Any], None]] = None,
    ) -> None:
        self.api_key = api_key
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.video_reference = video_reference
        self.max_fps = max_fps
        self.on_prediction = on_prediction

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._capture = None
        self._prev_gray = None

    @classmethod
    def init_with_workflow(
        cls,
        api_key: Optional[str] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        video_reference: Any = 0,
        max_fps: int = 30,
        on_prediction: Optional[Callable[[dict, Any], None]] = None,
    ) -> "InferencePipeline":
        return cls(api_key, workspace_name, workflow_id, video_reference, max_fps, on_prediction)

    # -- Frame source helpers -------------------------------------------------
    def _open_capture(self):
        if _cv2 is None:
            return False
        try:
            idx = self.video_reference
            # Treat integers as camera indices
            if isinstance(idx, int):
                print(f"[inference] Opening webcam index {idx}")
                cap = _cv2.VideoCapture(idx)
                if not cap.isOpened():
                    print(f"[inference] Failed to open webcam index {idx}")
                    cap.release()
                    return False
                self._capture = cap
                return True
            # If it's a string path, try to open it as a file/stream
            if isinstance(idx, str):
                cap = _cv2.VideoCapture(idx)
                if not cap.isOpened():
                    print(f"[inference] Failed to open video {idx}")
                    cap.release()
                    return False
                self._capture = cap
                return True
        except Exception:
            traceback.print_exc()
        return False

    def _read_frame_from_capture(self):
        assert self._capture is not None
        ret, frame = self._capture.read()
        if not ret:
            return None
        return frame

    def _generate_frame(self):
        """Fallback synthetic frame generator (used when no cv2 or capture)."""
        if np is None:
            return None
        h, w = 240, 320
        frame = (np.ones((h, w, 3), dtype="uint8") * 64)
        t = int(time.time() * 2) % w
        frame[:, t:t + 20, 0] = 255
        return frame

    # -- Detection helpers ---------------------------------------------------
    def detect_on_frame(self, frame: Any) -> Tuple[Any, List[Detection]]:
        """Run a lightweight detector on `frame`.

        Returns a tuple of (annotated_frame, detections) where detections are dicts
        with keys: label, score, box (x,y,w,h)
        """
        detections: List[Detection] = []
        out = frame.copy() if hasattr(frame, "copy") else frame

        # Prefer OpenCV-based detection if a full cv2 is available and has the functions
        try:
            has_cv2_detector = _cv2 is not None and hasattr(_cv2, 'HoughCircles')
        except Exception:
            has_cv2_detector = False

        if has_cv2_detector and np is not None:
            try:
                gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)

                # -- Ball detection: HoughCircles for small circular bright blobs
                blurred = _cv2.GaussianBlur(gray, (9, 9), 2)
                circles = _cv2.HoughCircles(
                    blurred,
                    _cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=20,
                    param1=50,
                    param2=30,
                    minRadius=3,
                    maxRadius=80,
                )
                if circles is not None:
                    circles = circles.round().astype("int")
                    for (x, y, r) in circles[0, :]:
                        x0, y0, w0, h0 = x - r, y - r, r * 2, r * 2
                        detections.append({"label": "ball", "score": 0.7, "box": [int(x0), int(y0), int(w0), int(h0)]})
                        _cv2.circle(out, (x, y), r, (0, 255, 255), 2)
                        _cv2.circle(out, (x, y), 2, (0, 0, 255), 3)

                # -- Player detection: simple motion-based contour detection
                if self._prev_gray is not None:
                    diff = _cv2.absdiff(self._prev_gray, gray)
                    _, th = _cv2.threshold(diff, 25, 255, _cv2.THRESH_BINARY)
                    # dilate to combine regions
                    kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (3, 3))
                    th = _cv2.dilate(th, kernel, iterations=2)
                    contours, _ = _cv2.findContours(th, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = _cv2.contourArea(cnt)
                        if area < 500:  # ignore small noisy contours
                            continue
                        x, y, w, h = _cv2.boundingRect(cnt)
                        detections.append({"label": "player", "score": 0.6, "box": [x, y, w, h]})
                        _cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        _cv2.putText(out, "player", (x, y - 6), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                self._prev_gray = gray

            except Exception:
                traceback.print_exc()

            return out, detections

        # -- Fallback lightweight numpy-based detection (no OpenCV required)
        if np is None:
            return out, detections

        try:
            # grayscale
            gray = (frame[..., 0].astype('float32') * 0.299 + frame[..., 1].astype('float32') * 0.587 + frame[..., 2].astype('float32') * 0.114).astype('uint8')

            # Ball detection: bright regions
            bright_mask = gray > 200
            if bright_mask.sum() > 20:
                ys, xs = np.nonzero(bright_mask)
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                w0, h0 = x1 - x0, y1 - y0
                area = bright_mask.sum()
                # Heuristic: small-ish bright blob -> ball
                if 10 <= w0 <= 200 and 10 <= h0 <= 200 and area < 2000:
                    cx = int((x0 + x1) / 2)
                    cy = int((y0 + y1) / 2)
                    detections.append({"label": "ball", "score": 0.6, "box": [int(x0), int(y0), int(w0), int(h0)]})
                    # annotate
                    try:
                        # draw simple rectangle with numpy
                        out = out.copy()
                        out[y0:y1, x0:x1] = np.clip(out[y0:y1, x0:x1] + np.array([0, 80, 80], dtype='uint8'), 0, 255)
                    except Exception:
                        pass

            # Player detection: motion-based using difference with previous frame
            if self._prev_gray is not None:
                diff = np.abs(self._prev_gray.astype('int16') - gray.astype('int16')).astype('uint8')
                th = diff > 25
                # coarse connected areas via bounding boxes of non-zero regions
                if th.sum() > 0:
                    ys, xs = np.nonzero(th)
                    x0, x1 = xs.min(), xs.max()
                    y0, y1 = ys.min(), ys.max()
                    w0, h0 = x1 - x0, y1 - y0
                    if w0 * h0 > 500:
                        detections.append({"label": "player", "score": 0.5, "box": [int(x0), int(y0), int(w0), int(h0)]})
                        try:
                            out = out.copy()
                            out[y0:y1, x0:x1] = np.clip(out[y0:y1, x0:x1] + np.array([0, 120, 0], dtype='uint8'), 0, 255)
                        except Exception:
                            pass

            self._prev_gray = gray
        except Exception:
            traceback.print_exc()

        return out, detections

    # -- Run loop ------------------------------------------------------------
    def _run_loop(self):
        try:
            # Try to open capture; if fails, fall back to generator
            opened = False
            if isinstance(self.video_reference, (int, str)) and _cv2 is not None:
                opened = self._open_capture()

            delay = 1.0 / max(1, int(self.max_fps))
            while not self._stop_event.is_set():
                if self._capture is not None:
                    frame = self._read_frame_from_capture()
                    if frame is None:
                        # Failed to read from capture, stop capture and fallback
                        try:
                            self._capture.release()
                        except Exception:
                            pass
                        self._capture = None
                        frame = self._generate_frame()
                else:
                    frame = self._generate_frame()

                result: Dict[str, Any] = {}
                if frame is not None:
                    annotated, detections = self.detect_on_frame(frame)
                    result["output_image"] = OutputImage(annotated)
                    result["detections"] = detections
                else:
                    result["output_image"] = None
                    result["detections"] = []

                if callable(self.on_prediction):
                    try:
                        self.on_prediction(result, frame)
                    except Exception:
                        traceback.print_exc()

                time.sleep(delay)
        except Exception:
            traceback.print_exc()

    def start(self) -> None:
        """Start the background processing thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the pipeline to stop and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the background thread to finish (or until timeout)."""
        if self._thread:
            self._thread.join(timeout=timeout)


__all__ = ["InferencePipeline"]
