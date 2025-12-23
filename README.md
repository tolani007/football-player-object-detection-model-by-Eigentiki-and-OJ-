# football-player-object-detection-model-by-Eigentiki-and-OJ-
This is a soccer tracking computer vision model that me and my boy trained in one day 🫩💜

## Running the example script locally 🔧

The example `eigentiki-OJ-soccer-orus.py` can run your local webcam and process
frames with a lightweight local detector when OpenCV and NumPy are installed.

Install the runtime dependencies (prefer a virtual environment):

- pip install --upgrade pip
- pip install numpy opencv-python Pillow

Then run:

```bash
python3 eigentiki-OJ-soccer-orus.py
```

Notes:
- The pipeline will attempt to open `video_reference=0` (your default webcam).
- If you want the pipeline to use your own trained model, replace or extend
  `InferencePipeline.detect_on_frame` in `inference/__init__.py` to call your
  model and return `detections` with `label`, `score`, and `box`.
- If your environment is controlled by your OS/distribution (PEP 668) and
  prevents pip installs, create a virtualenv: `python3 -m venv .venv && source .venv/bin/activate`
