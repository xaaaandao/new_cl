from PIL import Image, ImageOps
import numpy as np
import random
import os

class RandomCropWithValidation:
    def __init__(self, size, min_info_ratio=0.1):
        self.size = size
        self.min_info_ratio = min_info_ratio

    def format_image(self, img):
        width, height = img.size
    
        if width > self.size or height > self.size:
            left = (width - self.size) // 2
            top = (height - self.size) // 2
            right = left + self.size
            bottom = top + self.size
            img = img.crop((left, top, right, bottom))
        else:
            width, height = img.size
            pad_width = max(0, self.size - width)
            pad_height = max(0, self.size - height)
            
            padding = (
                pad_width // 2,  # Esquerda
                pad_height // 2,  # Superior
                pad_width - pad_width // 2,  # Direita
                pad_height - pad_height // 2  # Inferior
            )
            
            img = ImageOps.expand(img, padding, fill=(255, 255, 255))
        
        return img

    def find_crop(self, img_array):
        for idx in range(10):  # Limitar o número de tentativas
            x = random.randint(0, img_array.shape[1] - self.size)
            y = random.randint(0, img_array.shape[0] - self.size)
            
            crop = img_array[y:y + self.size, x:x + self.size]

            information = np.sum(np.all(crop < 255, axis=-1))
            total_pixels = crop.shape[0] * crop.shape[1]
            
            if information / total_pixels >= self.min_info_ratio:
                return Image.fromarray(crop)
            else:
                pass

        return None
    
    def __call__(self, img):

        return_full_img = self.find_crop(np.array(img))

        if return_full_img != None:
            return return_full_img

        bbox = ImageOps.invert(img.convert('RGB')).getbbox()
        trimmed = img.crop(bbox)

        if trimmed.size[0] < self.size or trimmed.size[1] < self.size:
            img_trimmed = self.format_image(trimmed)
            return img_trimmed

        return_crop_img = self.find_crop(np.array(trimmed))

        if return_crop_img != None:
            return return_crop_img

        img_trimmed = self.format_image(trimmed)
        return img_trimmed
        