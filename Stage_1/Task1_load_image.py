
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# Locate the Stage_1 directory
STAGE_DIR = Path(__file__).resolve().parent

# Input image and output directory
IMAGE_PATH = STAGE_DIR / "Images" / "0000000100.png"
RESULTS_DIR = STAGE_DIR / "results"

# Create results folder if it does not exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the image
image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

# Check whether the image was loaded successfully
if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}\n"
        "Check the file path and filename."
    )

# Get image information
height, width, channels = image.shape

print("Image loaded successfully!")
print(f"Filename: {IMAGE_PATH.name}")
print(f"Image width: {width} pixels")
print(f"Image height: {height} pixels")
print(f"Number of channels: {channels}")
print(f"Image shape: {image.shape}")
print(f"Data type: {image.dtype}")

# Save a copy of the loaded image
output_path = RESULTS_DIR / "0000000100_loaded.png"

if not cv2.imwrite(str(output_path), image):
    raise IOError(f"Could not save image: {output_path}")

print(f"Saved image to: {output_path}")

# OpenCV uses BGR; Matplotlib expects RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(12, 6))
plt.imshow(image_rgb)
plt.title("KITTI-360 Camera Image - Frame 0000000100")
plt.axis("off")
plt.tight_layout()
plt.show()
