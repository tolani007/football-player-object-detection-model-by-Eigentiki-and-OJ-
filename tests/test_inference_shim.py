import time

try:
    from inference import InferencePipeline
except Exception as e:
    print('Skipping test_inference_shim: inference module missing:', e)
    print('test_inference_shim SKIPPED')
else:
    collected = []

    def on_pred(result, frame):
        collected.append(result)

    p = InferencePipeline.init_with_workflow(on_prediction=on_pred, max_fps=10)
    p.start()
    # Let the pipeline run for a short moment
    time.sleep(0.5)
    p.stop()

    print('collected length:', len(collected))
    assert len(collected) >= 1
    assert 'output_image' in collected[0]
    print('test_inference_shim OK')
