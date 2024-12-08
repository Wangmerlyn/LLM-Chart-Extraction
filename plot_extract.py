import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def extract_and_save_plot_points(
    image_path,
    target_color,
    tolerance=10,
    link_distance=5,
    legend_filter_length=50,
    output_dir="output/plot_points",
    output_suffix="_plot_points",
    debug=False,
):
    """
    Extracts and links plot points in a chart image based on the specified color, removes legend samples,
    interpolates between segments, and saves the points as a sorted NumPy array.

    Parameters:
        image_path (str): Path to the input image.
        target_color (tuple): The RGB color of the plot to extract (e.g., (0, 255, 0) for green).
        tolerance (int): Tolerance for color matching.
        link_distance (int): Maximum distance between points to link fragments.
        legend_filter_length (int): Maximum segment length to be considered a legend sample.
        output_dir (str): Directory to save the extracted points as a CSV file.
        output_suffix (str): Suffix to append to the output file name.

    Returns:
        np.ndarray: A 2D NumPy array where each row represents a point (x, y), sorted by x.
    """
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for color matching
    lower_bound = np.array([max(0, c - tolerance) for c in target_color])
    upper_bound = np.array([min(255, c + tolerance) for c in target_color])

    # Create a mask for the target color
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    # Find the coordinates of the matching pixels
    y_coords, x_coords = np.where(mask > 0)
    plot_points = sorted(list(zip(x_coords, y_coords)), key=lambda p: p[0])

    if not plot_points:
        print("No points detected for the specified color.")
        return np.array([])

    # Link nearby points
    linked_points = []
    current_segment = [plot_points[0]]

    for i in range(1, len(plot_points)):
        if (
            cdist([current_segment[-1]], [plot_points[i]])[0][0]
            <= link_distance
        ):
            current_segment.append(plot_points[i])
        else:
            linked_points.append(current_segment)
            current_segment = [plot_points[i]]
    if current_segment:
        linked_points.append(current_segment)

    # Filter out potential legend samples
    filtered_segments = []
    for segment in linked_points:
        segment = np.array(segment)
        if len(segment) > 1:
            y_variance = np.var(segment[:, 1])
            x_span = segment[-1, 0] - segment[0, 0]
            if y_variance > 1 or x_span > legend_filter_length:
                filtered_segments.append(segment)

    # Combine points from segments
    final_points = []
    for segment in filtered_segments:
        final_points.extend([tuple(point) for point in segment])

    # Convert to NumPy array and sort by x-coordinate
    final_array = np.array(final_points)
    final_array = final_array[np.argsort(final_array[:, 0])]

    # Prepare output file name and path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file_name = f"{base_name}{output_suffix}.csv"
    output_path = os.path.join(output_dir, output_file_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the points to a CSV file
    np.savetxt(output_path, final_array, delimiter=",", fmt="%d")
    print(f"Extracted points saved to {output_path}")
    if not debug:
        return final_array
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.scatter(final_array[:, 0], final_array[:, 1], s=1, color="red")
    plt.gca().invert_yaxis()
    plt.title("Extracted Plot Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    return final_array


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
