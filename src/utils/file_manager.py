import os
import cv2

class FileManager:
    @staticmethod
    def ensure_directory(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def save_image(path, image, params=None):
        cv2.imwrite(path, image, params)
        
    @staticmethod
    def save_mask(path, mask):
        cv2.imwrite(path, mask)