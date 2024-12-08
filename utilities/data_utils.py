"""
This script contains utility functions for working with data.
"""

import base64
from mimetypes import guess_type


def local_image_to_data_url(image_path):
    """
    Convert a local image file to a data URL.
    """
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = (
            "application/octet-stream"  # Default MIME type if none is found
        )

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode(
            "utf-8"
        )

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"
