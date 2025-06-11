import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Define thresholds for classification
def classify_plant(features, rusty_area):
    """
    A rule-based classifier to distinguish Healthy, Unhealthy (Rusty), and Unhealthy Leaf (Dark).
    """
    color_mean = features['Color Mean (BGR)']
    edge_count = features['Edge Count']

    # Conditions for classification based on rusty area and other features
    if rusty_area > 0.15:  # At least 15% of the image is rusty (rusty red)
        return "Unhealthy (Rusty Leaf)"
    elif color_mean[1] > 120 and color_mean[1] > color_mean[2] + 30 and edge_count > 500:
        # Healthy: Green dominance, green > red by a margin, and sufficient edges
        return "Healthy (Green Leaf)"
    elif color_mean[2] > color_mean[0] + 50 and color_mean[2] > color_mean[1] + 50:
        # Dark red or brown (High red and low green/blue, indicating decay or rust)
        return "Unhealthy (Brown/Rusty Leaf)"
    elif edge_count < 300:  # Few edges, likely unhealthy
        return "Unhealthy (Damaged Leaf)"
    else:
        return "Unknown Condition"

def upload_image():
    """Opens a file dialog to upload an image."""
    Tk().withdraw()
    filename = askopenfilename(title="Select an Image File",
                                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return filename

def extract_features(image):
    """Extracts basic features from the plant image."""
    features = {}

    # Resize the image
    resized_image = cv2.resize(image, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Color features (mean and standard deviation in BGR channels)
    mean, std_dev = cv2.meanStdDev(resized_image)
    features['Color Mean (BGR)'] = mean.flatten().tolist()
    features['Color StdDev (BGR)'] = std_dev.flatten().tolist()

    # Shape features using edge detection
    edges = cv2.Canny(gray, 100, 200)
    features['Edge Count'] = int(np.sum(edges > 0))

    # Create a mask for rusty areas (high red, low green, low blue)
    rusty_mask = (resized_image[:, :, 2] > resized_image[:, :, 1] + 50) & (resized_image[:, :, 2] > resized_image[:, :, 0] + 50)
    rusty_area = np.sum(rusty_mask) / (128 * 128)  # Proportion of rusty pixels

    return features, resized_image, edges, rusty_area

def main():
    print("Upload a plant image to extract features...")
    file_path = upload_image()

    if not file_path:
        print("No file selected.")
        return

    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Could not read the image. Please upload a valid file.")
        return

    # Extract features
    features, resized_image, edges, rusty_area = extract_features(image)

    # Display extracted features
    print("\nExtracted Features:")
    for key, value in features.items():
        if isinstance(value, list):  # Handle lists
            print(f"{key}: {value[:3]}...")  # Show first 3 elements for lists
        else:
            print(f"{key}: {value}")  # Directly print scalar values

    # Classify the plant based on extracted features
    classification = classify_plant(features, rusty_area)
    print(f"\nPlant Health Status: {classification}")

    # Show images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
   
    plt.subplot(1, 2, 2)
    plt.title("Edges")
    plt.imshow(edges, cmap="gray")
    plt.axis("off")
   
    plt.suptitle(f"Classification: {classification}")
    plt.show()

if __name__ == "__main__":
    main()

