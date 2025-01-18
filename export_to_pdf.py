from PIL import Image
import os
import argparse

def create_pdf_from_images(input_folder, output_pdf, rows, cols, image_width=None):
    # A4 dimensions in pixels at 300 DPI
    A4_WIDTH, A4_HEIGHT = 2480, 3508

    # Get a list of all PNG images in the folder
    image_files = [
        os.path.join(input_folder, file)
        for file in os.listdir(input_folder)
        if file.lower().endswith(".png")
    ]

    if not image_files:
        raise ValueError("No PNG images found in the specified folder.")

    images = [Image.open(img).convert("RGBA") for img in image_files]

    # Resize images if width is specified
    if image_width:
        images = [
            img.resize((image_width, int(img.height * image_width / img.width)), Image.LANCZOS)
            for img in images
        ]

    # Ensure images have a white background
    images = [
        Image.alpha_composite(
            Image.new("RGBA", img.size, "white"), img
        ).convert("RGB") for img in images
    ]

    # Determine the size of each image (assuming all are the same after resizing)
    max_width, max_height = max(img.size for img in images)

    # Create a new page canvas with A4 dimensions
    page_width, page_height = A4_WIDTH, A4_HEIGHT

    # Calculate available grid space
    x_spacing = (page_width - (cols * max_width)) // (cols + 1)
    y_spacing = (page_height - (rows * max_height)) // (rows + 1)

    if x_spacing < 0 or y_spacing < 0:
        raise ValueError("Images are too large to fit in the specified grid on an A4 page.")

    pages = []
    canvas = Image.new("RGB", (page_width, page_height), "white")
    x_offset, y_offset = x_spacing, y_spacing
    image_count = 0

    for img in images:
        canvas.paste(img, (x_offset, y_offset))
        image_count += 1

        # Update offsets
        x_offset += max_width + x_spacing
        if image_count % cols == 0:
            x_offset = x_spacing
            y_offset += max_height + y_spacing

        # If the page is filled, save it and start a new page
        if image_count % (rows * cols) == 0:
            pages.append(canvas)
            canvas = Image.new("RGB", (page_width, page_height), "white")
            x_offset, y_offset = x_spacing, y_spacing

    # Add the last page if it's not already added
    if image_count % (rows * cols) != 0:
        pages.append(canvas)

    # Save the pages to the output PDF
    pages[0].save(output_pdf, save_all=True, append_images=pages[1:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a PDF from PNG images in a grid layout.")
    parser.add_argument("input_folder", help="Folder containing PNG images.")
    parser.add_argument("output_pdf", help="Output PDF file name.")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows per page.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns per page.")
    parser.add_argument("--image_width", type=int, default=None, help="Optional width to resize each image while maintaining the aspect ratio.")

    args = parser.parse_args()

    try:
        create_pdf_from_images(args.input_folder, args.output_pdf, args.rows, args.cols, args.image_width)
        print(f"PDF created successfully: {args.output_pdf}")
    except Exception as e:
        print(f"Error: {e}")
