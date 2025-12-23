import os
import cv2  # type: ignore
from dotenv import load_dotenv
from inference import InferencePipeline

# 1. Load the API key from the .env file
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

def my_sink(result, video_frame):
    # Display an image from the workflow response
    out_img = result.get("output_image")
    if out_img:
        try:
            cv2.imshow("Workflow Image", out_img.numpy_image)
            cv2.waitKey(1)
        except Exception:
            # in headless or shim environments imshow may be a no-op
            pass

    # Print a concise summary to avoid spamming the console
    dets = result.get("detections") or []
    if dets:
        print(f"Detections: {len(dets)} - {dets[:3]}")
    else:
        # only print when there are detections (silent otherwise)
        pass

# 2. Initialize a pipeline object using the environment variable
pipeline = InferencePipeline.init_with_workflow(
    api_key=api_key,
    workspace_name="eigentiki",
    workflow_id="detect-and-classify",
    video_reference=0, # 0 for webcam
    max_fps=30,
    on_prediction=my_sink
)

# 3. Start the pipeline with a user prompt so we "request" webcam access
import sys
import time

try:
    confirm = input("Press Enter to request webcam access and start the pipeline (or type 'n' to abort): ")
    if confirm.strip().lower() == 'n':
        print('Aborted by user.')
        sys.exit(0)
except KeyboardInterrupt:
    print('Aborted by user.')
    sys.exit(0)

pipeline.start()

# Wait briefly to see if the pipeline opened a webcam capture, otherwise fall back
for _ in range(30):
    if getattr(pipeline, '_capture', None) is not None:
        print('[main] Webcam opened by pipeline.')
        break
    time.sleep(0.1)
else:
    print('[main] Webcam not opened, using synthetic frames or no camera available.')

print("Press 'q' in the image window or Ctrl+C here to stop.")

# Main loop: keep process alive and check for 'q' key to quit
try:
    while True:
        # If cv2 is available, check for quit key in windows shown by the callback
        try:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print('[main] Quit requested via keypress.')
                break
        except Exception:
            # If waitKey is not functional (headless), rely on keyboard interrupt
            pass
        time.sleep(0.1)
except KeyboardInterrupt:
    print('\n[main] Interrupted by user (KeyboardInterrupt).')
finally:
    pipeline.stop()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print('[main] Pipeline stopped and windows closed.')
