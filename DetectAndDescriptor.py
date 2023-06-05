import cv2
import numpy as np


def detectAndDescribe_sift(image):
    sift = cv2.SIFT_create()
    (kps, features) = sift.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)