import os
import cv2
from dotenv import load_dotenv
from inference import InferencePipeline

# 1. Load the API key from the .env file
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

def my_sink(result, video_frame):
    # Display an image from the workflow response
    if result.get("output_image"): 
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    # Print predictions to the console
    print(result)

# 2. Initialize a pipeline object using the environment variable
pipeline = InferencePipeline.init_with_workflow(
    api_key=api_key,
    workspace_name="eigentiki",
    workflow_id="detect-and-classify",
    video_reference=0, # 0 for webcam
    max_fps=30,
    on_prediction=my_sink
)

# 3. Start the pipeline
pipeline.start()
pipeline.join()
