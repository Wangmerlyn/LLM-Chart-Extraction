import cv2
import numpy as np
import argparse
from collections import Counter


def get_top_n_colors(image_path, n=10):
    """
    Perform color frequency analysis on an image and return the top N most frequent colors.

    Parameters:
        image_path (str): Path to the input image.
        n (int): Number of top colors to return (default is 10).

    Returns:
        List of tuples: Each tuple contains a color (RGB) and its count in the image.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image into a 2D array where each row is an RGB value
    pixels = image_rgb.reshape(-1, 3)

    # Use Counter to count the frequency of each color (as a tuple)
    color_counts = Counter(map(tuple, pixels))

    # Get the top N most common colors
    top_colors = color_counts.most_common(n)

    return top_colors


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
