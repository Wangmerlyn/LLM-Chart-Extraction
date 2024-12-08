import argparse
from PIL import Image


def crop_image(image_path, top_left, bottom_right, output_path):
    """
    Crops an image based on the given bounding box.

    Parameters:
    - image_path (str): Path to the input image.
    - top_left (tuple): Coordinates of the top-left corner (x, y).
    - bottom_right (tuple): Coordinates of the bottom-right corner (x, y).
    - output_path (str): Path to save the cropped image.
    """
    # Open the image
    image = Image.open(image_path)

    # Crop the image using the bounding box
    cropped_image = image.crop((*top_left, *bottom_right))

    # Save the cropped image
    cropped_image.save(output_path)
    print(f"Cropped image saved to {output_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Crop an image using bounding box coordinates."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "top_left",
        type=str,
        help="Top-left corner coordinates (x, y) as 'x,y'.",
    )
    parser.add_argument(
        "bottom_right",
        type=str,
        help="Bottom-right corner coordinates (x, y) as 'x,y'.",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the cropped image."
    )

    args = parser.parse_args()

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

    # Call the crop function
    crop_image(args.image_path, top_left, bottom_right, args.output_path)


if __name__ == "__main__":
    main()

# python crop_img.py <image_path> <top_left> <bottom_right> <output_path>
