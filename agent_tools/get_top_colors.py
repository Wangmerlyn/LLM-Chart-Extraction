import cv2
import numpy as np
import argparse
from collections import Counter



import cv2
import numpy as np
from PIL import Image
from collections import Counter
from typing import Union, List, Tuple

def get_top_n_colors(image_input: Union[str, np.ndarray, Image.Image], n: int = 10) -> List[Tuple[Tuple[int, int, int], int]]:
    """
    Perform color frequency analysis on an image and return the top N most frequent colors.

    Parameters:
        image_input (str | np.ndarray | PIL.Image.Image): Path to the image, or image object.
        n (int): Number of top colors to return (default is 10).

    Returns:
        List[Tuple[Tuple[int, int, int], int]]: List of (RGB color, count) tuples.
    """
    # Handle image input types
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        # Assume it's OpenCV format (BGR), convert to RGB
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        # Convert PIL to RGB then to numpy array
        image_rgb = np.array(image_input.convert("RGB"))
    else:
        raise TypeError(f"image_input must be a file path, NumPy array, or PIL.Image.Image, got {type(image_input)} instead.")

    # Flatten the image to a list of RGB tuples
    pixels = image_rgb.reshape(-1, 3)

    # Count color frequencies
    color_counts = Counter(map(tuple, pixels))

    # Return top N
    return color_counts.most_common(n)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Get the most frequent colors from an image"
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top colors to display (default is 10).",
    )

    return parser.parse_args()


def main():
    # Parse the arguments
    args = parse_args()

    # Get the top N colors from the image
    top_colors = get_top_n_colors(args.image_path, args.top_n)

    # Print the top N most frequent colors and their counts
    for idx, (color, count) in enumerate(top_colors, 1):
        print(f"Color {idx}: RGB = {color}, Count = {count}")


if __name__ == "__main__":
    main()
