from __future__ import annotations

from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional


class Image:
    def __init__(self, img: np.ndarray) -> None:
        self.image = img

    @property
    def image(self) -> np.ndarray:
        return self._image
    
    @image.setter
    def image(self, img) -> None:
        if len(img.shape) == 2:
            self._image = img

        elif len(img.shape) == 3:
            self._image = np.empty(img.shape[:2])
            self._image[:, :] = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3    

        self.shape = self._image.shape
        self.height = self.shape[0]
        self.width = self.shape[1]

    @classmethod
    def from_path(cls, path: str) -> Image:
        return cls(img=plt.imread(path))
    
    def resize(self, height: int, width: Optional[int]=None) -> Image:
        height_factor = height / self.height

        if width == None:
            width = int(self.width * height_factor)
        width_factor = width / self.width
        
        old_w_coord = np.arange(0, self.width)*width_factor
        old_h_coord = np.arange(0, self.height)*height_factor
        
        f = interp2d(old_w_coord, old_h_coord, self.image)
        
        new_w_coord = np.arange(0, width)
        new_h_coord = np.arange(0, height)
        new_image = f(new_w_coord, new_h_coord)

        return self.__class__(new_image)

    def transpose(self) -> Image:
        return self.__class__(np.flip(self.image))
    
    def show(self, dpi: int=60) -> None:
        plt.figure(figsize=(self.width/dpi, self.height/dpi))
        plt.axis('off')
        plt.imshow(self.image, cmap='Greys')
        plt.show()


class Kernel(Image):
    def __init__(self, img: np.ndarray) -> None:
        super().__init__(img)