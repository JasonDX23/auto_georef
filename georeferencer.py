import cv2
import os
import numpy as np
import re
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds

def extract_black_pixels(image_path, tolerance=10):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Please check the path.")

    # Define the black color range
    lower_bound = np.array([0, 0, 0])  # Black lower bound
    upper_bound = np.array([tolerance, tolerance, tolerance])  # Black upper bound

    # Create a mask for black pixels
    binary_mask = cv2.inRange(image, lower_bound, upper_bound)

    return binary_mask

def detect_long_lines(binary_mask, min_line_length_ratio=0.25, threshold=50, max_line_gap=10):
    # Ensure the input is binary
    if len(binary_mask.shape) != 2:
        raise ValueError("Input mask must be a binary image.")

    # Image dimensions
    height, width = binary_mask.shape

    # Minimum line length in pixels
    min_line_length = int(width * min_line_length_ratio)

    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        binary_mask,
        rho=1, theta=np.pi / 180, threshold=threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    # Create a blank image to draw lines
    output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return lines, output_image

def separate_horizontal_vertical(lines, angular_tolerance=5):
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        angle = abs(angle)

        if abs(angle) <= angular_tolerance or abs(angle - 180) <= angular_tolerance:
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(angle - 90) <= angular_tolerance:
            vertical_lines.append((x1, y1, x2, y2))

    return horizontal_lines, vertical_lines


def find_outer_frame(horizontal_lines, vertical_lines, height, width):
    # Sort horizontal lines by their y-coordinates
    horizontal_lines.sort(key=lambda line: (line[1] + line[3]) // 2)
    top_line = horizontal_lines[0]
    bottom_line = horizontal_lines[-1]

    # Sort vertical lines by their x-coordinates
    vertical_lines.sort(key=lambda line: (line[0] + line[2]) // 2)
    left_line = vertical_lines[0]
    right_line = vertical_lines[-1]

    # Approximate bottom line
    bottom_limit = height * 0.9
    bottom_line_candidates = [line for line in horizontal_lines if (line[1] + line[3]) // 2 < bottom_limit]
    if bottom_line_candidates:
        bottom_line = bottom_line_candidates[-1]

    # Find intersections
    corners = [
        (left_line[0], top_line[1]),  # Top-left
        (right_line[2], top_line[1]),  # Top-right
        (left_line[0], bottom_line[3]),  # Bottom-left
        (right_line[2], bottom_line[3])  # Bottom-right
    ]

    return corners

def refine_corners(binary_image, corners, kernel_size=5, roi_size=15):
    # Create a Sobel kernel for edge detection
    sobel_x = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(binary_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    refined_corners = []
    for corner in corners:
        x, y = corner
        x, y = int(x), int(y)

        # Define ROI
        x_start, x_end = max(0, x - roi_size // 2), min(binary_image.shape[1], x + roi_size // 2)
        y_start, y_end = max(0, y - roi_size // 2), min(binary_image.shape[0], y + roi_size // 2)

        # Extract ROI from gradient magnitude
        roi = gradient_magnitude[y_start:y_end, x_start:x_end]

        # Find the pixel with the highest intensity within the ROI
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
        refined_x = x_start + max_loc[0]
        refined_y = y_start + max_loc[1]

        refined_corners.append((refined_x, refined_y))

    return refined_corners

def calculate_raster_resolution(top_line, bottom_line):
    top_y = (top_line[1] + top_line[3]) // 2
    bottom_y = (bottom_line[1] + bottom_line[3]) // 2
    pixel_distance = abs(bottom_y - top_y)
    return pixel_distance / 391


def estimate_inner_corners(outer_corners, resolution, frame_width_mm=12):
    offset = frame_width_mm * resolution
    inner_corners = [
        (x + offset, y + offset) if i in [0] else
        (x - offset, y + offset) if i in [1] else
        (x + offset, y - offset) if i in [2] else
        (x - offset, y - offset)
        for i, (x, y) in enumerate(outer_corners)
    ]
    return inner_corners


def refine_inner_corners(binary_image, estimated_corners, kernel_size=5, roi_size=15):
    return refine_corners(binary_image, estimated_corners, kernel_size, roi_size)

def format_filename(filename):
    # Regex pattern to extract the relevant parts
    pattern = r"(\d{2} [A-Z])\](\d{2})"
    match = re.search(pattern, filename)
    if match:
        # Format the result as '47A/14' and remove leading zero from the second part
        main_part = match.group(1).replace(' ', '')  # Remove space
        number_part = str(int(match.group(2)))  # Convert to int to remove leading zero
        return f"{main_part}/{number_part}"
    else:
        return None

image = input("Please enter image path: ")
if (image.startswith("'") and image.endswith("'")) or (image.startswith('"') and image.endswith('"')):
    image = image[1:-1]

black_mask = extract_black_pixels(image, tolerance=150)
cv2.imwrite("outputs/black_pixels_mask.png", black_mask)
binary_mask = cv2.imread(r"outputs\black_pixels_mask.png", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if binary_mask is None:
    raise ValueError("Failed to load the binary mask image. Check the file path.")

lines, output_image = detect_long_lines(binary_mask, min_line_length_ratio=0.25)

cv2.imwrite("outputs/lines_detected.jpg", output_image)

binary_mask = cv2.imread(r"outputs\black_pixels_mask.png", cv2.IMREAD_GRAYSCALE)
lines, output_image = detect_long_lines(binary_mask, min_line_length_ratio=0.25)

# Separate horizontal and vertical lines
horizontal_lines, vertical_lines = separate_horizontal_vertical(lines, angular_tolerance=5)

# Find the initial corners
height, width = binary_mask.shape
initial_corners = find_outer_frame(horizontal_lines, vertical_lines, height, width)

# Refine the corner positions
refined_corners = refine_corners(binary_mask, initial_corners, kernel_size=5, roi_size=15)

# Draw the refined corners on the output image
for x, y in refined_corners:
    cv2.circle(output_image, (int(x), int(y)), 5, (255, 0, 0), -1)

cv2.imwrite(r"outputs\refined_corners.png", output_image)

binary_mask = cv2.imread(r"outputs\black_pixels_mask.png", cv2.IMREAD_GRAYSCALE)

# Detect frame lines and corners
lines, output_image = detect_long_lines(binary_mask, min_line_length_ratio=0.25)
horizontal_lines, vertical_lines = separate_horizontal_vertical(lines, angular_tolerance=5)
height, width = binary_mask.shape
outer_corners = find_outer_frame(horizontal_lines, vertical_lines, height, width)

# Calculate raster resolution
top_line = horizontal_lines[0]
bottom_line = horizontal_lines[-1]
resolution = calculate_raster_resolution(top_line, bottom_line)

# Estimate inner corners
estimated_inner_corners = estimate_inner_corners(outer_corners, resolution)

# Refine inner corners
refined_inner_corners = refine_inner_corners(binary_mask, estimated_inner_corners)

# Draw outer and inner corners on the output image
for x, y in outer_corners:
    cv2.circle(output_image, (int(x), int(y)), 5, (255, 0, 0), -1)  # Outer corners
for x, y in refined_inner_corners:
    cv2.circle(output_image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Refined inner corners

cv2.imwrite(r"outputs\outer_and_inner_corners.png", output_image)

original_image = cv2.imread(image)
corners_array = np.array(refined_inner_corners)

# Calculate the bounding box of the inner corners
x_min, y_min = corners_array.min(axis=0)
x_max, y_max = corners_array.max(axis=0)

# Crop the image using the bounding box
cropped_image = original_image[int(y_min):int(y_max), int(x_min):int(x_max)]

cv2.imwrite(r"outputs\cropped_image.png", cropped_image)

filename = image
formatted = format_filename(filename)
print("Formatted:", formatted)

# Read the GeoJSON file into a GeoDataFrame
gdf = gpd.read_file('soi_osm_sheet_index.geojson.json')

# Check if 'formatted' exists in 'EVEREST_SH' and retrieve geometry
if formatted in gdf['EVEREST_SH'].values:
    geometry = gdf.loc[gdf['EVEREST_SH'] == formatted, 'geometry'].values[0]
    print("Geometry found:", geometry)

    # Load the cropped image
    cropped_image = cv2.imread(r"outputs\cropped_image.png")
    height, width = cropped_image.shape[:2]

    # Get the bounding box of the geometry
    minx, miny, maxx, maxy = geometry.bounds

    # Calculate affine transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Define CRS (coordinate reference system)
    crs = "epsg:4326"  # Assuming WGS84, modify if needed

    # Save as a GeoTIFF
    cropped_image_path = "outputs/cropped_image.png"
    sanitized_formatted = formatted.replace("/", "_")  # Replace invalid characters
    output_tiff = os.path.join("outputs", f"{sanitized_formatted}.tif")
    with rasterio.open(cropped_image_path, 'r+') as ds:
        with rasterio.open(output_tiff, 'w', driver='GTiff',
                           height=ds.height, width=ds.width,
                           count=ds.count, dtype=cropped_image.dtype,
                           crs=crs, transform=transform) as dst:
            for i in range(1, ds.count + 1):
                dst.write(ds.read(i), i)

    print("GeoTIFF created at:", output_tiff)
else:
    print("Formatted text not found in EVEREST_SH")