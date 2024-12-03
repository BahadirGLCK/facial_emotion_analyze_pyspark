import os
import requests

class ImageDownloader:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def download_image(self, url: str, image_name: str, person_dir: str) -> str:
        # Ensure the person's folder exists
        os.makedirs(person_dir, exist_ok=True)

        # Define the save path for the image
        save_path = os.path.join(person_dir, image_name)

        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return "success"
            else:
                return f"failed: HTTP {response.status_code}"
        except Exception as e:
            return f"failed: {e}"