import cv2
import dlib
import argparse
from PIL import Image
import numpy as np
import os
import random
from pillow_heif import register_heif_opener
from math import atan2, degrees
from multiprocessing import Pool

# Register HEIF/HEIC image format
register_heif_opener()

def process_image(args):
    filename, output_folder, shape_predictor_path, hat_images, final_width = args
    try:
        # Load dlib's face detector and shape predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor_path)

        # Open image using OpenCV
        image = Image.open(filename).convert("RGBA")
        image_np = np.array(image)  # Convert to NumPy array for OpenCV
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)  # Convert to grayscale

        (h, w) = (new_w, new_h) = image_cv.shape[:2]
        rotation = 0

        while rotation < 270:
            faces = detector(image_cv)
            if len(faces) == 0:
                rotation += 90
                print(f"No faces found in {filename}, try {rotation} rotation")
            else:
                break

            rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -90, 1.0)

            # Calculate the new dimensions of the rotated image
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_w = int(h * sin_angle + w * cos_angle)
            new_h = int(h * cos_angle + w * sin_angle)

            # Adjust the rotation matrix to account for translation
            rotation_matrix[0, 2] += (new_w - w) / 2
            rotation_matrix[1, 2] += (new_h - h) / 2

            # Perform the rotation with adjusted bounds
            image_cv = cv2.warpAffine(image_cv, rotation_matrix, (new_w, new_h))
            opencv_img = np.array(image)
            opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
            rot_image = cv2.warpAffine(
                opencv_img, rotation_matrix, (new_w, new_h)
            )
            rot_image = cv2.cvtColor(rot_image, cv2.COLOR_BGR2RGB)
            # Convert NumPy array to PIL Image
            image = Image.fromarray(rot_image)

            # cv2.imshow("Debug Visualization", image_cv)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Process each detected face
        for i, face in enumerate(faces):
            # Get facial landmarks
            shape = predictor(image_cv, face)
            landmarks = np.array(
                [(shape.part(j).x, shape.part(j).y) for j in range(68)]
            )

            # Get coordinates of the eyes
            left_eye = landmarks[36:42].mean(axis=0)  # Average of left eye landmarks
            right_eye = landmarks[42:48].mean(axis=0)  # Average of right eye landmarks

            # Calculate the angle to rotate the face
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            angle = degrees(atan2(delta_y, delta_x))  # Angle in degrees
            (h, w) = image_cv.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D(
                (w // 2, h // 2), angle, 1.0
            )
            # Perform the rotation with adjusted bounds
            image_cv = cv2.warpAffine(image_cv, rotation_matrix, (new_w, new_h))
            opencv_img = np.array(image)
            opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
            rot_image = cv2.warpAffine(
                opencv_img, rotation_matrix, (new_w, new_h)
            )
            rot_image = cv2.cvtColor(rot_image, cv2.COLOR_BGR2RGB)
            # Convert NumPy array to PIL Image
            image = Image.fromarray(rot_image)

            # Compute again landmarks
            rotated_faces = detector(image_cv)
            if len(rotated_faces) == 0:
                print(f"No face detected in rotated image for {filename}")
                continue
            rotated_face = rotated_faces[0]
            rotated_shape = predictor(image_cv, rotated_face)
            rotated_landmarks = np.array([(rotated_shape.part(j).x, rotated_shape.part(j).y) for j in range(68)])

            # Include ears by extending the jawline points outward
            extended_landmarks = list(rotated_landmarks[:17])  # Jawline points (0–16)

            # Add forehead extension
            forehead_height = int(abs(rotated_landmarks[27][1] - rotated_landmarks[8][1]) * 0.6)
            forehead_points = [
                (x, y - forehead_height) for (x, y) in rotated_landmarks[:17]
            ]
            extended_landmarks.extend(forehead_points[::-1])

            # Create a convex hull
            hull = cv2.convexHull(np.array(extended_landmarks))

            # Create a mask for the face
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            cv2.fillPoly(mask, [hull], 255)

            # Apply the mask to the image
            mask_pil = Image.fromarray(mask, mode="L")

            cropped_face = Image.new("RGBA", image.size, (0, 0, 0, 0))
            cropped_face.paste(image, (0, 0), mask=mask_pil)

            # Crop to the bounding box of the mask
            bbox = mask_pil.getbbox()
            cropped_face = cropped_face.crop(bbox)

            # Add a birthday hat
            hat = random.choice(hat_images)  # Randomly select a hat
            hat_width = int(cropped_face.width * 1.1)  # Scale the hat width relative to the face
            hat_height = int(hat_width * hat.height / hat.width)  # Maintain aspect ratio
            resized_hat = hat.resize((hat_width, hat_height), Image.LANCZOS)

            # Expand the canvas to make room for the hat
            overlap_hat_height = int(0.10 * cropped_face.height)
            expanded_height = cropped_face.height + hat_height -  overlap_hat_height  # Add extra space for the hat
            expanded_canvas = Image.new("RGBA", (hat_width, expanded_height), (0, 0, 0, 0))
            expanded_canvas.paste(cropped_face, (int((cropped_face.width * 1.1 - cropped_face.width) / 2), hat_height - overlap_hat_height))

            # Position the hat above the face
            # hat_x = (expanded_canvas.width - resized_hat.width) // 2
            hat_x = 0
            hat_y = 0
            expanded_canvas.paste(resized_hat, (hat_x, hat_y), mask=resized_hat)

            # Resize the final image
            expanded_canvas = expanded_canvas.resize((final_width, int(final_width * expanded_canvas.height / expanded_canvas.width)), Image.LANCZOS)

            # Save the cropped face
            output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}_face_{i + 1}.png"
            output_path = os.path.join(output_folder, output_filename)
            expanded_canvas.save(output_path, "PNG")
            print(f"Saved cropped face to {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

def crop_face_with_smooth_mask(input_folder, output_folder, shape_predictor_path, hat_folder, width):
    # Load all hat images
    hat_images = [Image.open(os.path.join(hat_folder, hat)) for hat in os.listdir(hat_folder) if hat.endswith(".png")]
    if not hat_images:
        raise ValueError("No hat images found in the specified folder.")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]

    args = [(file, output_folder, shape_predictor_path, hat_images, width) for file in files]

    # Use ThreadPoolExecutor for parallel processing
    with Pool() as pool:
        pool.map(process_image, args)
        pool.close()  # Close the pool to prevent more tasks from being added
        pool.join()   # Wait for all worker processes to finish


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates tag images of faces wearing a birthday hat")
    parser.add_argument("input", help="Path to the folder containing the input images")
    parser.add_argument("--output", default="output", help="Path to the output folder")
    parser.add_argument("--resize-width", default=500, type=int, help="Width of the final image")
    args = parser.parse_args()
    shape_predictor_path = (
        "shape_predictor_68_face_landmarks.dat"  # Path to dlib's shape predictor model
    )
    hat_folder = "hats"  # Folder containing birthday hats (PNG format)

    # Download shape predictor if not already downloaded
    if not os.path.exists(shape_predictor_path):
        import urllib.request

        print("Downloading shape predictor model...")
        url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = shape_predictor_path + ".bz2"
        urllib.request.urlretrieve(url, compressed_path)
        import bz2

        with bz2.BZ2File(compressed_path) as bz2file, open(
            shape_predictor_path, "wb"
        ) as outfile:
            outfile.write(bz2file.read())
        os.remove(compressed_path)

    # Run the script
    crop_face_with_smooth_mask(args.input, args.output, shape_predictor_path, hat_folder, args.resize_width)
