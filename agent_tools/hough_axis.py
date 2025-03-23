import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple

'''
detect_axis_bounding_box is useful but not dedicated to the axis bounding box detection.
detect_axis_bounding_box_contour is more dedicated to the axis bounding box detection but cound result in slightly bigger boxes.
'''

def detect_axis_bounding_box(image_input:Union[np.ndarray,str,Image.Image], output_dir:Optional[str]=None, output_suffix:Optional[str]=None, debug=False,**kwargs):
    """
    Detects the bounding box of the axis frame in a chart image using Hough Line Transform.

    Parameters:
        image_input (str | np.ndarray | PIL.Image.Image): Path to image, or image object.
        output_dir (str): Directory to save the bounding box CSV file.
        output_suffix (str): Suffix to append to the output file name.
        debug (bool): If True, display the image with bounding box.

    Returns:
        tuple: Coordinates of the bounding box ((min_x, min_y), (max_x, max_y)) or None if not detected.
    """

    # === Handle input type ===
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        base_name = os.path.splitext(os.path.basename(image_input))[0]
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
        base_name = "image_array"
    elif isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        base_name = "pil_image"
    else:
        raise TypeError("image_input must be a file path, NumPy array, or PIL.Image.Image")

    if image is None:
        raise ValueError("Failed to load image.")

    # === Preprocessing ===
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # === Detect lines using Hough Transform ===
    # check if the parameters are in kwargs
    threshold = kwargs.get("threshold", 80)
    minLineLength = kwargs.get("minLineLength", 100)
    maxLineGap = kwargs.get("maxLineGap", 1)
    
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap
    )

    if lines is None:
        raise ValueError("No lines detected. Check the image or parameters.")

    # === Separate horizontal and vertical lines ===
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:  # horizontal
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(x1 - x2) < 10:  # vertical
            vertical_lines.append((x1, y1, x2, y2))

    if not horizontal_lines or not vertical_lines:
        raise ValueError("No horizontal or vertical lines detected. Check the image or parameters.")

    # === Bounding box coordinates ===
    min_y = min(line[1] for line in horizontal_lines)
    max_y = max(line[1] for line in horizontal_lines)
    min_x = min(line[0] for line in vertical_lines)
    max_x = max(line[0] for line in vertical_lines)
    bounding_box = ((min_x, min_y), (max_x, max_y))

    # === Save bounding box ===
    if output_dir is not None and output_suffix is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_name = f"{base_name}_{output_suffix}.csv"
        output_path = os.path.join(output_dir, output_file_name)
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(output_path, bounding_box, delimiter=",", fmt="%d")
        print(f"Bounding box coordinates saved to {output_path}")

    # === Optional visualization ===
    if debug:
        output_image = image.copy()
        cv2.rectangle(output_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Box for Axis Frame")
        plt.axis("off")
        plt.show()

    return bounding_box

def detect_axis_bounding_box_contour(image_input, debug=False) -> Optional[Tuple[Tuple[int, int]]]:
    """
    Detects the 4 corner points of the coordinate axis box in a structured chart image.

    Parameters:
        image_input (str or np.ndarray): Path to image or image loaded with cv2.

    Returns:
        List of 4 (x, y) tuples representing corners in order: top-left, top-right, bottom-right, bottom-left
    """
    # Load image
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    elif isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Input must be a file path or cv2 image, got {type(image_input)} instead.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Optional: blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binary threshold or Canny
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    # Or: thresh = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return None

    # Find the largest contour assuming it's the axis box
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    axis_contour = contours[0]

    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(axis_contour, True)
    approx = cv2.approxPolyDP(axis_contour, epsilon, True)

    if len(approx) != 4:
        print("Could not detect 4 corners accurately. Approximated points:", len(approx))
        return None

    # Extract 4 corner points
    pts = approx.reshape((4, 2))

    # Sort corners: top-left, top-right, bottom-right, bottom-left
    def sort_corners(pts):
        pts = sorted(pts, key=lambda x: x[1])  # sort by y first
        top = sorted(pts[:2], key=lambda x: x[0])   # top-left, top-right
        bottom = sorted(pts[2:], key=lambda x: x[0], reverse=True)  # bottom-right, bottom-left
        return [tuple(top[0]), tuple(top[1]), tuple(bottom[0]), tuple(bottom[1])]

    ordered_corners = sort_corners(pts)
    # return in the shape of  ((99, 11), (642, 198))
    ordered_corners = (ordered_corners[0], ordered_corners[2])

    # === Optional visualization ===
    if debug:
        min_x, min_y = ordered_corners[0]
        max_x, max_y = ordered_corners[1]
        output_image = image.copy()
        cv2.rectangle(output_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title("Bounding Box for Axis Frame")
        plt.axis("off")
        plt.show()
    return ordered_corners


def parse_args():
    """
    Parses command-line arguments for the script.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect axis bounding box in a chart image."
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to the input image."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the bounding box CSV file.",
        default="output/borderline",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        help="Suffix to append to the output file name.",
        default="_bounding_box",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display the detected bounding box on the image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detect_axis_bounding_box(
        args.image_path, args.output_dir, args.output_suffix, args.debug
    )


if __name__ == "__main__":
    main()
