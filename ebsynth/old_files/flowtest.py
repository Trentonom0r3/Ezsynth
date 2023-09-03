from smoothing import smooth_flow
import cv2
import matplotlib.pyplot as plt
import numpy as np

def flow_to_color(flow):
    # Calculate the angle and magnitude of the flow
    angle = np.arctan2(flow[..., 1], flow[..., 0]) + np.pi
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # Normalize angle and magnitude
    angle = (angle / (2 * np.pi)) * 255
    magnitude = (magnitude / np.max(magnitude)) * 255

    # Convert to HSV color space
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = angle
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = magnitude

    # Convert HSV to BGR color space
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr

# Paths
INPUT_SEQ = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input"
STYLE_SEQ = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Output"

# Initialize the smooth_flow class
smoother = smooth_flow(INPUT_SEQ, STYLE_SEQ, flow_method="RAFT")

# Smooth the style sequence
smoothed_stylized_sequence = smoother.apply_guided_warping()


# Visualize the smoothed stylized sequence
for i, frame in enumerate(smoothed_stylized_sequence):
    cv2.imwrite(f"C:/Users/tjerf/Desktop/Testing/src/Testvids/Output/smoothed_stylized_sequence_{i}.png", frame)
