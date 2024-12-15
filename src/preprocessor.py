import os
import cv2

class ImagePreprocessor:
    def __init__(self, size=(128, 128)):
        self.size = size

    def preprocess_image_and_save(self, image_path: str, bbox: str, save_dir:str, save_name: str, person_name: str) -> str:
        """
        Crop and resize the image based on bbox, then save it in the processed directory.
        """
        # Parse bbox coordinates
        try:
            x1, y1, x2, y2 = map(int, bbox.split(","))
        except Exception as e:
            return f"Failed to parse bbox {bbox}: {e}"

        # Load the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                return f"Image not found: {image_path}"
        except Exception as e:
            return f"Failed to load image {image_path}: {e}"

        # Crop and resize the image
        try:
            cropped = image[y1:y2, x1:x2]
            resized = cv2.resize(cropped, self.size)
        except Exception as e:
            return f"Failed to preprocess image {image_path}: {e}"

        # Save the processed image
        person_dir = os.path.join(save_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        save_path = os.path.join(person_dir, save_name)

        try:
            cv2.imwrite(save_path, resized)
            return f"Processed and saved to {save_path}"
        except Exception as e:
            return f"Failed to save processed image {save_path}: {e}"
        
    def preprocess_image(self, image_path: str, bbox: str) -> tuple:
        """
        Crop and resize the image based on bbox, returning the preprocessed image as a NumPy array.
        """
        # Parse bbox coordinates
        try:
            x1, y1, x2, y2 = map(int, bbox.split(","))
        except Exception as e:
            return None, f"Failed to parse bbox {bbox}: {e}"

        # Load the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, f"Image not found: {image_path}"
        except Exception as e:
            return None, f"Failed to load image {image_path}: {e}"

        # Crop and resize the image
        try:
            cropped = image[y1:y2, x1:x2]
            resized = cv2.resize(cropped, self.size)
            return resized, "processed"
        except Exception as e:
            return None, f"Failed to preprocess image {image_path}: {e}"