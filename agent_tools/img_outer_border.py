"""
This script adds custom ticks and a border to the input image and optionally saves the image.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def get_image_with_border(
    image_path, tick_pixel=50, output_dir=None, output_suffix="with_border"
):
    """
    Adds custom ticks and a border to the input image and optionally saves the image.

    Parameters:
        image_path (str): The path to the image to be displayed and annotated.
        tick_pixel (int): The interval (in pixels) for setting custom ticks on the axes.
        output_dir (str, optional): The directory to save the resulting image. If None, the image is not saved.
        output_suffix (str): The suffix to append to the output filename before the extension.

    Returns:
        None: The function shows the image with borders and ticks, and optionally saves it.
    """
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get the height and width of the image
    height, width, _ = np.array(img_rgb).shape

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set custom ticks every `tick_pixel` pixels
    x_ticks = range(0, width, tick_pixel)
    y_ticks = range(0, height, tick_pixel)

    # Set the ticks on the x and y axes
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Rotate x-axis labels to make them readable
    plt.xticks(rotation=90)

    # Add grid lines for better visualization (optional)
    ax.grid(visible=True, linestyle="-", color="gray", alpha=0.5)

    # Display the image
    ax.imshow(img_rgb)

    # Save the image if an output directory is provided
    if output_dir:
        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Extract the base name of the image file (without extension)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Create the output file name with the suffix and the same file extension as the original image
        output_filename = (
            f"{base_name}_{output_suffix}{os.path.splitext(image_path)[1]}"
        )
        output_path = os.path.join(output_dir, output_filename)

        # Save the image with tight bounding box and no padding
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        print(f"Image saved at: {output_path}")

    # Show the image
    return output_path


# Example usage
# get_image_with_border("your_image_path.jpg", tick_pixel=50, output_dir="test_figs/given", output_suffix="with_border")