import cv2
import numpy as np
from scipy.signal import find_peaks

# Function for R peak detection
def detect_peaks(signal, threshold):
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks

# Read ECG image
ecg_image = cv2.imread('full.png', cv2.IMREAD_GRAYSCALE)

# Preprocessing
ecg_image_blurred = cv2.GaussianBlur(ecg_image, (7, 7), 0)

# Edge detection and segmentation
edges = cv2.Canny(ecg_image_blurred, threshold1=50, threshold2=150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour corresponds to the ECG waveform
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the ECG waveform
mask = np.zeros_like(ecg_image)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# R peak detection
vertical_projection = np.sum(mask, axis=0)
r_peak_indices = detect_peaks(vertical_projection, threshold=150)

# Beat extraction
beat_width = 120 # Adjust as needed
beats = []
for r_peak_index in r_peak_indices:
    # Ensure symmetric extraction around the R-peak
    start_index = max(0, r_peak_index - beat_width // 2)
    end_index = min(r_peak_index + beat_width // 2, len(mask[0]))
    beat_roi = mask[:, start_index:end_index]
    beats.append(beat_roi)

# Artifact removal and quality assessment
# You can implement artifact removal algorithms here
# Example: Remove beats with low amplitude or irregular shape

# Display the original image and extracted beats for visualization
cv2.imshow('Original ECG Image', ecg_image)
for i, beat in enumerate(beats):
    cv2.imshow(f'Extracted Beat {i+1}', beat)

cv2.waitKey(0)
cv2.destroyAllWindows()
