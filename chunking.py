import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class OCRRecognitionPreprocessor:
    def __init__(self, target_height=32, target_width=804):
        self.target_height = target_height
        self.target_width = target_width

    def keepratio_resize(self, img):
        try:
            if img is None or len(img.shape) < 2:
                raise ValueError("Invalid image input")

            cur_ratio = img.shape[1] / float(img.shape[0])
            mask_height = self.target_height
            mask_width = self.target_width

            if cur_ratio > float(mask_width) / mask_height:
                cur_target_height = self.target_height
                cur_target_width = self.target_width
            else:
                cur_target_height = self.target_height
                cur_target_width = int(self.target_height * cur_ratio)

            img = cv2.resize(img, (cur_target_width, cur_target_height))

            # Handle both grayscale and color images
            if len(img.shape) == 2:  # Grayscale
                mask = np.zeros([mask_height, mask_width], dtype=img.dtype)
                mask[:img.shape[0], :img.shape[1]] = img
            else:  # Color
                mask = np.zeros([mask_height, mask_width, img.shape[2]], dtype=img.dtype)
                mask[:img.shape[0], :img.shape[1], :] = img

            return mask
        except Exception as e:
            print(f"Error in keepratio_resize: {e}")
            return None

    def chunk_image(self, img):
        # Chunk the image into 3 parts
        chunks = [img[:, (300 - 48) * i : (300 - 48) * i + 300] for i in range(3)]
        return chunks

    def process_and_save_chunks(self, input_dir, output_dir):
        """Process all images in the input directory and save their chunks to the output directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image file extensions
                image_path = os.path.join(input_dir, filename)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Error loading image {filename}")
                    continue

                resized_img = self.keepratio_resize(img)
                if resized_img is None:
                    continue

                chunks = self.chunk_image(resized_img)
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"{os.path.splitext(filename)[0]}_chunk{i}.png"
                    chunk_path = os.path.join(output_dir, chunk_filename)

                    # Convert chunk to PIL for saving
                    chunk_pil = Image.fromarray(chunk)
                    chunk_pil.save(chunk_path)

                print(f"Processed {filename} and saved chunks.")

    def __call__(self, inputs):
        # Existing __call__ method code ...
        pass

# Example usage
input_directory = "data/chuncktest"
output_directory = "data/chunckvalidate"

preprocessor = OCRRecognitionPreprocessor()
preprocessor.process_and_save_chunks(input_directory, output_directory)
