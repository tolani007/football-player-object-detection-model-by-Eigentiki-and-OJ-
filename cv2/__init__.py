"""Lightweight local shim for `cv2` to allow repository code to import `cv2` even when
OpenCV is not installed in the environment.

Behavior:
- If a real `cv2` package is importable, this module forwards attribute access to it.
- Otherwise it provides minimal, safe implementations of common functions used in
  this repo (`imshow`, `waitKey`, `namedWindow`, `destroyAllWindows`).

This shim is intentionally minimal: it tries to display images in notebooks (via
IPython) when available, otherwise it saves images to the temporary directory and
prints the file path so the developer can inspect the output when running headless.
"""

from __future__ import annotations

import importlib
import importlib.util
import importlib.machinery
import sys
import os
import tempfile
import time
from typing import Any

_real_cv2 = None

# Attempt to locate a compiled OpenCV extension and load it without re-importing
# this shim (which would cause recursion). We search sys.path for cv2 extension
# files (e.g., cv2*.so) and load the first candidate as a module named
# "_cv2_impl".
def _find_and_load_real_cv2() -> Any:
    # If a different cv2 is already loaded (not this module), use it
    existing = sys.modules.get("cv2")
    if existing is not None and existing is not sys.modules.get(__name__):
        return existing

    for p in sys.path:
        if not p:
            continue
        # Candidate locations
        cand_dir = os.path.join(p, "cv2")
        if os.path.isdir(cand_dir):
            for fname in os.listdir(cand_dir):
                # look for extension modules (Linux .so, macOS .so, Windows .pyd)
                if fname.startswith("cv2") and (fname.endswith(".so") or fname.endswith(".pyd") or ".so." in fname):
                    path = os.path.join(cand_dir, fname)
                    try:
                        spec = importlib.util.spec_from_file_location("_cv2_impl", path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)  # type: ignore
                            return module
                    except Exception:
                        # try next candidate
                        continue
        # Also consider top-level extension like cv2.so in site-packages root
        for ext in (".so", ".pyd"):
            top_path = os.path.join(p, f"cv2{ext}")
            if os.path.isfile(top_path):
                try:
                    spec = importlib.util.spec_from_file_location("_cv2_impl", top_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)  # type: ignore
                        return module
                except Exception:
                    continue
    return None

try:
    _real_cv2 = _find_and_load_real_cv2()
except Exception:
    _real_cv2 = None


# If the real cv2 exists, forward attribute access to it
if _real_cv2 is not None:
    def __getattr__(name: str) -> Any:  # type: ignore
        # guard: don't delegate back to this module
        if sys.modules.get("cv2") is sys.modules.get(__name__):
            return getattr(_real_cv2, name)
        return getattr(_real_cv2, name)

    def __dir__() -> list[str]:  # type: ignore
        return sorted(list(globals().keys()) + dir(_real_cv2))

else:
    # Minimal shim implementations used by this repo.
    # Provide a lightweight VideoCapture implementation using ffmpeg when OpenCV
    # is not available. This allows the example to access a webcam device such as
    # /dev/video0 on Linux without requiring full OpenCV installation.
    import shutil
    import subprocess
    from io import BytesIO

    _FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

    class VideoCapture:
        def __init__(self, source=0):
            # map integer indices to /dev/video{idx} on Linux if possible
            self.source = source
            if isinstance(source, int):
                self.device = f"/dev/video{source}"
            else:
                self.device = str(source)
            self._opened = False
            # Consider device opened if it exists and ffmpeg is present
            if _FFMPEG_AVAILABLE and os.path.exists(self.device):
                self._opened = True
            else:
                # attempt to open via ffmpeg probe to check availability
                if _FFMPEG_AVAILABLE:
                    try:
                        subprocess.check_call(["ffmpeg", "-f", "v4l2", "-list_formats", "all", "-i", self.device], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self._opened = True
                    except Exception:
                        self._opened = False
                else:
                    self._opened = False

        def isOpened(self):
            return self._opened

        def read(self):
            # Capture a single frame using ffmpeg and return (ret, frame)
            if not self._opened:
                return False, None
            if not _FFMPEG_AVAILABLE:
                return False, None
            try:
                # ffmpeg: grab one frame and output as jpeg to stdout
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "v4l2",
                    "-i",
                    self.device,
                    "-vframes",
                    "1",
                    "-f",
                    "image2",
                    "-"
                ]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                out, _ = p.communicate(timeout=5)
                if not out:
                    return False, None
                try:
                    from PIL import Image
                    import numpy as _np
                    buf = BytesIO(out)
                    img = Image.open(buf).convert('RGB')
                    arr = _np.asarray(img)
                    # convert RGB PIL to BGR numpy (OpenCV convention)
                    arr = arr[:, :, ::-1].copy()
                    return True, arr
                except Exception:
                    return False, None
            except Exception:
                return False, None

        def release(self):
            self._opened = False

    def namedWindow(name: str, flags: int | None = None) -> None:
        """Create a named window (no-op in headless shim)."""
        # No-op for headless environments
        return None

    def destroyAllWindows() -> None:
        """Destroy any created windows (no-op in headless shim)."""
        return None

    def imshow(window_name: str, image: Any) -> None:
        """Display or persist an image in headless environments.

        Behavior:
        - If running in a Jupyter environment and Pillow is available, display inline.
        - Else, if Pillow is available, save PNG to the system temp directory and print
          the path so it can be inspected.
        - Otherwise print a short message saying we could not render the image.
        """
        # Lazy imports to avoid hard dependencies
        pil_available = False
        np = None
        try:
            from PIL import Image
            pil_available = True
        except Exception:
            pil_available = False

        try:
            import numpy as _np
            np = _np
        except Exception:
            np = None

        img = None
        # Convert numpy arrays to PIL Image when possible
        if pil_available:
            from PIL import Image
            if hasattr(image, "__array__") and np is not None:
                arr = np.asarray(image)
                # Handle common shapes
                try:
                    if arr.ndim == 2:
                        img = Image.fromarray(arr.astype("uint8"), mode="L")
                    elif arr.ndim == 3 and arr.shape[2] == 3:
                        img = Image.fromarray(arr.astype("uint8"))
                    elif arr.ndim == 3 and arr.shape[2] == 4:
                        img = Image.fromarray(arr.astype("uint8"))
                    else:
                        # fallback
                        img = Image.fromarray(arr.astype("uint8"))
                except Exception:
                    # last resort: attempt to construct image without mode
                    try:
                        img = Image.fromarray(arr)
                    except Exception:
                        img = None
            elif isinstance(image, Image.Image):
                img = image

        # Try to display inline in notebook
        displayed = False
        if img is not None:
            try:
                # IPython display (works in Jupyter)
                from IPython.display import display, Image as IPImage
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                display(IPImage(data=buf.getvalue(), format="png"))
                displayed = True
            except Exception:
                displayed = False

        if not displayed and img is not None:
            # Save to temp and report path
            try:
                tmp_dir = tempfile.gettempdir()
                safe_name = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in window_name)[:200]
                tmp_path = os.path.join(tmp_dir, f"cv2_shim_{safe_name}.png")
                img.save(tmp_path)
                print(f"[cv2 shim] Saved image to: {tmp_path}")
                return
            except Exception:
                pass

        # If we cannot do anything helpful, print a short fallback message
        print("[cv2 shim] imshow called but image could not be rendered; install Pillow or import real cv2 for full functionality.")

    def waitKey(delay: int = 0) -> int:
        """Wait for a key event for given milliseconds. In the shim, we sleep and return -1."""
        try:
            if delay > 0:
                time.sleep(delay / 1000.0)
        except Exception:
            pass
        return -1

    # Provide a helpful module-level attribute to indicate this is the shim
    __is_shim__ = True

__all__ = ["imshow", "waitKey", "namedWindow", "destroyAllWindows", "VideoCapture"]
