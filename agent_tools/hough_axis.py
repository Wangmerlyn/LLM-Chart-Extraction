import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_axis_bounding_box(image_path):
    """
    Detects the bounding box of the axis frame in a chart image using Hough Line Transform.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        tuple: Coordinates of the bounding box ((min_x, min_y), (max_x, max_y)) or None if not detected.
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10
    )

    if lines is None:
        print("No lines detected.")
        return None

    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:  # Horizontal line
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(x1 - x2) < 10:  # Vertical line
            vertical_lines.append((x1, y1, x2, y2))

    # Ensure valid lines are found
    if not horizontal_lines or not vertical_lines:
        print("Insufficient lines detected for bounding box.")
        return None

    # Find the bounding box coordinates
    min_y = min(line[1] for line in horizontal_lines)
    max_y = max(line[1] for line in horizontal_lines)
    min_x = min(line[0] for line in vertical_lines)
    max_x = max(line[0] for line in vertical_lines)
    bounding_box = ((min_x, min_y), (max_x, max_y))

    # Visualize the result
    output_image = image.copy()
    cv2.rectangle(output_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Box for Axis Frame")
    plt.axis("off")
    plt.show()

    return bounding_box


# Example usage
image_path = "test_figs/given/plot_0_1.png"
bounding_box = detect_axis_bounding_box(image_path)
bounding_box
# save bounding box to a csv file
np.savetxt(
    "test_figs/given/plot_0_1_bounding_box.csv", bounding_box, delimiter=","
)
