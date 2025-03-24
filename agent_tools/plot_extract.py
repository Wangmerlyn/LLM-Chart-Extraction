import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Union, Optional, Tuple
import pytesseract

def extract_and_save_plot_points(
    image_input: Union[str, np.ndarray, Image.Image],
    target_color,
    tolerance=30,
    link_distance=10,
    legend_filter_length=50,
    output_dir="output/plot_points",
    output_suffix="_plot_points",
    debug=False,
):
    """
    Extracts and links plot points in a chart image based on the specified color, removes legend samples,
    interpolates between segments, and saves the points as a sorted NumPy array.

    Parameters:
        image_input (str | np.ndarray | PIL.Image.Image): Input image.
        target_color (tuple): The RGB color of the plot to extract (e.g., (0, 255, 0) for green).
        tolerance (int): Tolerance for color matching.
        link_distance (int): Maximum distance between points to link fragments.
        legend_filter_length (int): Maximum segment length to be considered a legend sample.
        output_dir (str): Directory to save the extracted points as a CSV file.
        output_suffix (str): Suffix to append to the output file name.
        debug (bool): Whether to visualize the result.

    Returns:
        np.ndarray: A 2D NumPy array where each row is a point (x, y), sorted by x.
    """

    # === Load and normalize input image ===
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Failed to read image from path: {image_input}")
        base_name = os.path.splitext(os.path.basename(image_input))[0]
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
        base_name = "image_array"
    elif isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        base_name = "pil_image"
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}. Must be str, np.ndarray, or PIL.Image.Image.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === Create mask for target color ===
    lower_bound = np.array([max(0, c - tolerance) for c in target_color])
    upper_bound = np.array([min(255, c + tolerance) for c in target_color])
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    y_coords, x_coords = np.where(mask > 0)
    plot_points = sorted(list(zip(x_coords, y_coords)), key=lambda p: p[0])

    if not plot_points:
        print("No points detected for the specified color.")
        return np.array([])

    # === Link nearby points ===
    linked_points = []
    current_segment = [plot_points[0]]
    for i in range(1, len(plot_points)):
        if cdist([current_segment[-1]], [plot_points[i]])[0][0] <= link_distance:
            current_segment.append(plot_points[i])
        else:
            linked_points.append(current_segment)
            current_segment = [plot_points[i]]
    if current_segment:
        linked_points.append(current_segment)

    # === Filter out legend samples ===
    filtered_segments = []
    for segment in linked_points:
        segment = np.array(segment)
        if len(segment) > 1:
            y_variance = np.var(segment[:, 1])
            x_span = segment[-1, 0] - segment[0, 0]
            if y_variance > 1 or x_span > legend_filter_length:
                filtered_segments.append(segment)

    # === Combine final points ===
    final_points = []
    for segment in filtered_segments:
        final_points.extend([tuple(point) for point in segment])

    if not final_points:
        print("All segments filtered out.")
        return np.array([])

    plot_array = np.array(final_points)
    plot_array = plot_array[np.argsort(plot_array[:, 0])]

    # === Group by x and average y ===
    final_array = np.array(
        [
            [x, np.mean(plot_array[plot_array[:, 0] == x][:, 1])]
            for x in np.unique(plot_array[:, 0])
        ]
    )

    # === Save to file ===
    if output_dir is not None and output_suffix is not None:
        output_file_name = f"{base_name}{output_suffix}.csv"
        output_path = os.path.join(output_dir, output_file_name)
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(output_path, final_array, delimiter=",", fmt="%d")
        print(f"Extracted points saved to {output_path}")

    # === Debug visualization ===
    if debug:
        plt.figure(figsize=(12, 6))
        plt.scatter(plot_array[:, 0], plot_array[:, 1], s=1, color="red")
        plt.gca().invert_yaxis()
        plt.title("Extracted Plot Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    return final_array

def mask_alphabetic_text(image_input: Union[str, np.ndarray, Image.Image],
                         config: str = "") -> np.ndarray:
    """
    Performs OCR detection on an input image to find alphabetic text,
    and masks them off by drawing black rectangles over the detected text.

    Parameters:
        image_input (str | np.ndarray | PIL.Image.Image):
            The input image. Can be a file path, an OpenCV image (numpy.ndarray), or a PIL Image.
        output_path (str):
            The file path where the masked image will be saved.
        config (str, optional):
            Additional configuration options to pass to pytesseract (default is an empty string).

    Returns:
        np.ndarray: The masked image (in BGR format, as used by OpenCV).
    """
    # --- Load image based on input type ---
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Failed to load image from {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    elif isinstance(image_input, Image.Image):
        # Convert PIL image to OpenCV (BGR) format.
        image = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("image_input must be a file path, np.ndarray, or PIL.Image.Image.")

    # --- Convert image to grayscale for OCR ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Optionally, apply pre-processing (thresholding, blur, etc.) for better OCR performance.
    # For example: gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # --- Perform OCR and get detailed data including bounding boxes ---
    ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=config)
    n_boxes = len(ocr_data['level'])

    # --- Iterate over detected text and mask alphabetic characters ---
    for i in range(n_boxes):
        text = ocr_data['text'][i].strip()
        if text and text.isalpha():  # Only process alphabetic text
            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                          ocr_data['width'][i], ocr_data['height'][i])
            # Draw a filled rectangle (white) over the detected text area
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)
    # return the image as a PIL.Image.Image
    # Convert back to RGB for PIL compatibility
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def parse_args():
    """
    Parse command-line arguments for the script.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract plot points from a chart image."
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to the input image."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the extracted points as a CSV file.",
        default="output/plot_points",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        help="Suffix to append to the output file name.",
        default="_plot_points",
    )
    parser.add_argument(
        "--target_color",
        type=str,
        required=True,
        help="Target RGB color in the format 'R,G,B'.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=10,
        help="Tolerance for color matching (default: 10).",
    )
    parser.add_argument(
        "--link_distance",
        type=int,
        default=5,
        help="Maximum distance to link fragments (default: 5).",
    )
    parser.add_argument(
        "--legend_filter_length",
        type=int,
        default=50,
        help="Maximum segment length to identify legend lines (default: 50).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to visualize the extracted points.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_color = tuple(map(int, args.target_color.split(",")))
    extract_and_save_plot_points(
        args.image_path,
        target_color,
        args.tolerance,
        args.link_distance,
        args.legend_filter_length,
        args.output_dir,
        args.output_suffix,
        args.debug,
    )


if __name__ == "__main__":
    main()
