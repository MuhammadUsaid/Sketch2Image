
from pathlib import Path

import cv2
import numpy as np


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "sketches"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_SIZE = 512


def preprocess_sketch(image_path, output_path):
    """
    Preprocess a sketch for sketch-to-image generation.

    Steps:
    1. Load image
    2. Convert to grayscale
    3. Normalize background and foreground
    4. Apply thresholding
    5. Resize while preserving aspect ratio
    6. Add white padding
    7. Save processed image
    """

    # Load image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Handle images with dark backgrounds
    if np.mean(gray) < 127:
        gray = 255 - gray

    # Reduce small image noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Convert sketch into black lines on a white background
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Preserve aspect ratio while resizing
    height, width = binary.shape

    scale = min(
        IMAGE_SIZE / width,
        IMAGE_SIZE / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        binary,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Create a white square canvas
    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE),
        255,
        dtype=np.uint8
    )

    # Center the sketch
    x_offset = (IMAGE_SIZE - new_width) // 2
    y_offset = (IMAGE_SIZE - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    # Save processed image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)

    print(f"Processed: {image_path.name}")
    print(f"Saved to: {output_path}")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    supported_formats = {".png", ".jpg", ".jpeg"}

    sketch_files = [
        file for file in INPUT_DIR.iterdir()
        if file.suffix.lower() in supported_formats
    ]

    if not sketch_files:
        print("No sketches found in data/sketches/")
        return

    for sketch_path in sketch_files:

        output_path = OUTPUT_DIR / f"{sketch_path.stem}.png"

        preprocess_sketch(
            sketch_path,
            output_path
        )

    print("\nPreprocessing completed!")


if __name__ == "__main__":
    main()
