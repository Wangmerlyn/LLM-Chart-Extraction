import os
import argparse
from typing import Union, Tuple, Optional
from PIL import Image
import numpy as np
import cv2


def crop_image(
    image_input: Union[str, np.ndarray, Image.Image],
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    output_dir: Optional[str]=None,
    output_suffix: Optional[str]=None,
) -> Union[str, Image.Image]:
    """
    Crops an image based on the given bounding box.

    Parameters:
    - image_input (str | np.ndarray | PIL.Image.Image): Input image (path, cv2 array, or PIL image).
    - top_left (tuple): Coordinates of the top-left corner (x, y).
    - bottom_right (tuple): Coordinates of the bottom-right corner (x, y).
    - output_dir (str): Directory to save the cropped image.
    - output_suffix (str): Suffix to append to the original file name.

    Returns:
    - str: Path to the cropped image.
    """

    # Handle input image
    if isinstance(image_input, str):
        image = Image.open(image_input)
        base_name, ext = os.path.splitext(os.path.basename(image_input))
    elif isinstance(image_input, np.ndarray):
        # Convert OpenCV BGR image to PIL RGB
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        base_name = "image_array"
        ext = ".png"
    elif isinstance(image_input, Image.Image):
        image = image_input
        base_name = "pil_image"
        ext = ".png"
    else:
        raise TypeError(
            f"Unsupported image_input type: {type(image_input)}. Expected str, np.ndarray, or PIL.Image.Image."
        )

    # Crop the image using the bounding box
    cropped_image = image.crop((*top_left, *bottom_right))

    # Generate output file name and path
    if output_suffix is not None and output_dir is not None:
        output_file_name = f"{base_name}{output_suffix}{ext}"
        output_path = os.path.join(output_dir, output_file_name)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the cropped image
        cropped_image.save(output_path)
        print(f"Cropped image saved to {output_path}")
        return output_path
    else:
        return cropped_image


def parse_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Crop an image using bounding box coordinates."
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to the input image."
    )
    parser.add_argument(
        "--top_left",
        type=str,
        help="Top-left corner coordinates (x, y) as 'x,y'.",
    )
    parser.add_argument(
        "--bottom_right",
        type=str,
        help="Bottom-right corner coordinates (x, y) as 'x,y'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the cropped image.",
        default="output/cropped",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        help="Suffix to append to the original file name.",
        default="_cropped",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # Parse coordinates
    try:
        top_left = tuple(map(int, args.top_left.split(",")))
        bottom_right = tuple(map(int, args.bottom_right.split(",")))
    except ValueError:
        print(
            "Error: Coordinates must be integers and provided in the format 'x,y'."
        )
        return

    # Ensure valid bounding box
    if len(top_left) != 2 or len(bottom_right) != 2:
        print(
            "Error: Both top-left and bottom-right coordinates must have two values (x, y)."
        )
        return

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Call the crop function
    crop_image(
        args.image_path,
        top_left,
        bottom_right,
        args.output_dir,
        args.output_suffix,
    )


if __name__ == "__main__":
    main()
