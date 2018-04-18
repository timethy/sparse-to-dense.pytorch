import numpy as np
import cv2

def simulate_matching(img, max):
    """Simulates stereo matching. Applies canny edge detection on an img within the mask and than dilatates"""
    # We do not use cv2.Canny, since that applies non max suppression
    # Take the gradient
    # Threshold

    # Dilatate