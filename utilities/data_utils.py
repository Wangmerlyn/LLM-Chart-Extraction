"""
This script contains utility functions for working with data.
"""

import base64
import io
from mimetypes import guess_type
import numpy as np

import PIL.Image
from PIL import Image

def image_to_data_url(image_or_path) -> str:
    """
    Convert either a local image file (by path) or a PIL.Image.Image
    to a base64-encoded data URL string.

    :param image_or_path: A string (file path) or a PIL.Image.Image object.
    :return: Data URL (str) in the form "data:<mime_type>;base64,<encoded_data>"
    """
    if isinstance(image_or_path, str):
        # Case 1: Input is a file path
        mime_type, _ = guess_type(image_or_path)
        if mime_type is None:
            mime_type = "application/octet-stream"  # Default fallback MIME type

        # Read image bytes and encode to base64
        with open(image_or_path, "rb") as f:
            base64_encoded_data = base64.b64encode(f.read()).decode("utf-8")

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    elif isinstance(image_or_path, Image.Image):
        # Case 2: Input is a PIL Image object

        # Use the image's format if available; default to PNG
        fmt = image_or_path.format if image_or_path.format else "PNG"

        # Build the appropriate MIME type (e.g., image/png, image/jpeg)
        mime_type = f"image/{fmt.lower()}"

        # Save the image to an in-memory buffer
        buffer = io.BytesIO()
        image_or_path.save(buffer, format=fmt)
        buffer.seek(0)

        # Encode the image bytes to base64
        base64_encoded_data = base64.b64encode(buffer.read()).decode("utf-8")

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    elif isinstance(image_or_path, np.ndarray):
        # Case 3: Input is a numpy array
        # Convert the numpy array to a PIL Image
        image = Image.fromarray(image_or_path)

        # Use the image's format if available; default to PNG
        fmt = image.format if image.format else "PNG"

        # Build the appropriate MIME type (e.g., image/png, image/jpeg)
        mime_type = f"image/{fmt.lower()}"

        # Save the image to an in-memory buffer
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        buffer.seek(0)

        # Encode the image bytes to base64
        base64_encoded_data = base64.b64encode(buffer.read()).decode("utf-8")

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    else:
        # Invalid input type
        raise TypeError(
            f"image_to_data_url expects either a file path (str) or a PIL.Image.Image object or numpy array, got {type(image_or_path)} instead."
        )

def image_path_to_image(image_path: str) -> PIL.Image.Image:
    """
    Convert a local image file path to a PIL.Image.Image object.
    Parameters:
        image_path (str): The path to the image file.
    """
    # Open the image file and convert it to RGB mode
    image = PIL.Image.open(image_path).convert("RGB")
    return image
