from __future__ import annotations

from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional


IMAGE_MODES = {'RGBA': 4,
               'RGB': 3,
               'GRAY': 1}

IMAGE_FORMATS = {'float32': 1,
                 'uint8': 255}


class Image:
    def __init__(self, img: np.ndarray, mode: str='RGBA') -> None:
        height = img.shape[0]
        width = img.shape[1]
        h_axis = np.linspace(0, 1, height)
        w_axis = np.linspace(0, 1, width)

        if str(img.dtype) in IMAGE_FORMATS.keys():
            new_img = img/IMAGE_FORMATS[str(img.dtype)]
        else:
            raise TypeError(f'Unexpected image format passed to Image: {img.dtype}')

        if img.ndim == 2:
            self._contentR = interp2d(w_axis, h_axis, new_img)
            self._contentG = self._contentR
            self._contentB = self._contentR
            self._contentA = lambda w, h: 0

        elif img.ndim == 3 and img.shape[2] == 3:
            self._contentR = interp2d(w_axis, h_axis, new_img[:, :, 0])
            self._contentG = interp2d(w_axis, h_axis, new_img[:, :, 1])
            self._contentB = interp2d(w_axis, h_axis, new_img[:, :, 2])
            self._contentA = lambda w, h: 0

        elif img.ndim == 3 and img.shape[2] == 4:
            self._contentR = interp2d(w_axis, h_axis, new_img[:, :, 0])
            self._contentG = interp2d(w_axis, h_axis, new_img[:, :, 1])
            self._contentB = interp2d(w_axis, h_axis, new_img[:, :, 2])
            self._contentA = interp2d(w_axis, h_axis, new_img[:, :, 3])

        else:
            raise TypeError(f'ndarray passed to Image is of unexpected shape: {img.shape}, \
                            should be (*, *), (*, *, 3) or (*, *, 4).')

        self.mode = mode
        self.shape = img.shape[:2]

    @property
    def mode(self) -> str:
        return self._mode
    
    @mode.setter
    def mode(self, mode: str) -> None:
        if not mode in IMAGE_MODES.keys():
            raise ValueError(f'Image received unexpected value for mode: {mode}, \
                             should be in {IMAGE_MODES.keys()}')
        
        self._mode = mode
        self._number_of_channels = IMAGE_MODES[mode]

    @property
    def shape(self) -> tuple:
        return self._shape
    
    @shape.setter
    def shape(self, shape: int | tuple) -> None:
        if type(shape) == int:
            height = shape
            width = int(self.shape[1] * height/self.shape[0])
        else:
            height = shape[0]
            width = shape[1]

        self._shape = (height, width)

    @classmethod
    def from_path(cls, path: str) -> Image:
        return cls(img=plt.imread(path))
    
    def as_array(self) -> np.ndarray:
        height = self.shape[0]
        width = self.shape[1]
        h_axis = np.linspace(0, 1, height)
        w_axis = np.linspace(0, 1, width)

        if self.mode == 'RGBA':
            arr = np.empty((*self.shape, 4))
            arr[:, :, 0] = self._contentR(w_axis, h_axis)
            arr[:, :, 1] = self._contentG(w_axis, h_axis)
            arr[:, :, 2] = self._contentB(w_axis, h_axis)
            arr[:, :, 3] = self._contentA(w_axis, h_axis)

        if self.mode == 'RGB':
            arr = np.empty((*self.shape, 3))
            arr[:, :, 0] = self._contentR(w_axis, h_axis)
            arr[:, :, 1] = self._contentG(w_axis, h_axis)
            arr[:, :, 2] = self._contentB(w_axis, h_axis)

        if self.mode == 'GRAY':
            arr = np.zeros(self.shape)
            arr += self._contentR(w_axis, h_axis)
            arr += self._contentG(w_axis, h_axis)
            arr += self._contentB(w_axis, h_axis)
            arr /= 3

        return arr

    def flip(self) -> None:
        self._contentR = lambda w, h: self._contentR(-w, -h)
        self._contentG = lambda w, h: self._contentG(-w, -h)
        self._contentB = lambda w, h: self._contentB(-w, -h)
        self._contentA = lambda w, h: self._contentA(-w, -h)
    
    def show(self, dpi: int=60) -> None:
        plt.figure(figsize=(self.shape[0]/dpi, self.shape[1]/dpi))
        plt.axis('off')
        if self.mode == 'GRAY':
            plt.imshow(self.as_array(), cmap='Greys')
        else:
            plt.imshow(self.as_array())
        plt.show()


class Kernel(Image):
    def __init__(self, img: np.ndarray) -> None:
        super().__init__(img)