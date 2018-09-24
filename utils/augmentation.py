from imgaug import augmenters as iaa
from skimage.color import rgb2gray
import operator
import numpy as np

DEFAULT_PROBS = {
    "fliplr": 0.5,
    "flipud": 0.3,
    "scale": 0.1,
    "scale_px": (0.98, 1.02),
    "translate": 0.15,
    "translate_perc": (-0.05, 0.05),
    "rotate": 0.2,
    "rotate_angle": (-5, 5)
}

def cropCenter(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def toGrayscale(img):
    if len(img.shape) >= 3 and img.shape[-1] == 3:
        img = rgb2gray(img)

    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)

    return img

class Augmentation:

    def __init__(self, probs=DEFAULT_PROBS):
        self.seq = iaa.Sequential([
            # iaa.Lambda(self._normalize, None),
            iaa.Fliplr(probs['fliplr']),
            iaa.Flipud(probs['flipud']),
            iaa.Sometimes(probs["scale"], iaa.Affine(scale={"x": probs['scale_px'], "y": probs['scale_px']})),
            iaa.Sometimes(probs["translate"], iaa.Affine(translate_percent={"x": probs['translate_perc'], "y": probs['translate_perc']})),
            iaa.Sometimes(probs["rotate"], iaa.Affine(rotate=probs["rotate_angle"]))
            ])

    def __call__(self, imgs):
        if isinstance(imgs, list):
            imgs = [self.seq.augment_images(img) for img in imgs]
        else:
            imgs = self.seq.augment_image(imgs)

        return imgs

    def refresh_random_state(self):
        self.seq.to_deterministic()
