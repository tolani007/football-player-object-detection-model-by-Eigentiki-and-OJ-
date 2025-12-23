try:
    import numpy as np
    import cv2
except Exception as e:
    print('Skipping test_inference_local_detection: numpy or cv2 missing:', e)
    print('test_inference_local_detection SKIPPED')
else:
    from inference import InferencePipeline

    # Create a synthetic frame containing a white circle (ball) and a rectangle (player)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # ball
    _center = (160, 120)
    cv2.circle(frame, _center, 10, (255, 255, 255), -1)
    # player rectangle (simulate motion with a filled rect)
    cv2.rectangle(frame, (20, 40), (70, 170), (100, 100, 100), -1)

    p = InferencePipeline.init_with_workflow(video_reference=None, max_fps=10)
    annotated, detections = p.detect_on_frame(frame)

    print('Found detections:', detections)
    assert any(d['label'] == 'ball' for d in detections), 'Ball not detected'
    assert any(d['label'] == 'player' for d in detections), 'Player not detected'
    print('test_inference_local_detection OK')
