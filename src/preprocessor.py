from src.utils.logger import get_logger
import os
import cv2

class ImagePreprocessor:
    def __init__(self, save_dir: str, size=(128, 128)):
        self.logger = get_logger("ImagePreprocessor")  # Initialize logger
        self.save_dir = save_dir
        self.size = size
        os.makedirs(self.save_dir, exist_ok=True)

    def preprocess_image(self, image_path: str, bbox: str, save_name: str) -> str:
        try:
            x1, y1, x2, y2 = map(int, bbox.split(","))
            image = cv2.imread(image_path)
            if image is not None:
                cropped = image[y1:y2, x1:x2]
                resized = cv2.resize(cropped, self.size)
                save_path = os.path.join(self.save_dir, save_name)
                cv2.imwrite(save_path, resized)
                self.logger.info(f"Processed {save_name} with bbox {bbox}")
                return "processed"
            else:
                self.logger.warning(f"Image not found: {image_path}")
                return "image not found"
        except Exception as e:
            self.logger.error(f"Failed to process {save_name}: {e}")
            return f"failed: {e}"