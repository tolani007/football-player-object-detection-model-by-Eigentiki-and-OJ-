try:
    import numpy as np
    import cv2
except Exception as e:
    print('Skipping test_cv2_shim: numpy or cv2 missing:', e)
    print('test_cv2_shim SKIPPED')
else:
    print('cv2 module:', cv2)
    print('has imshow:', hasattr(cv2, 'imshow'))

    # Create a small random image and call imshow + waitKey
    arr = (np.random.rand(120, 160, 3) * 255).astype('uint8')
    cv2.imshow('test_window', arr)
    print('waitKey returns:', cv2.waitKey(10))
