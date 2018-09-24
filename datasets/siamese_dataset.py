from torchvision import transforms
from torch.utils.data import Dataset
from skimage.io import imread
from utils.augmentation import Augmentation, cropCenter, toGrayscale
from utils.visualisation_helpers import plot_side_by_side

import numpy as np
import os

get_idx = lambda s: "".join(filter(lambda x: x.isdigit(), s))

AUG_PROBS = {
    "fliplr": 0.4,
    "flipud": 0,
    "scale": 0,
    "scale_px": (1.0, 1.0),
    "translate": 0,
    "translate_perc": (0.0, 0.0),
    "rotate": 0,
    "rotate_angle": (-5, 5)
}

class SiameseDataset(Dataset):
    def __init__(self, config):
        super(SiameseDataset, self).__init__()

        func = []

        if config.augment:
            self.augmentor = Augmentation(probs=AUG_PROBS)
            func.append(transforms.Lambda(lambda img: self.augmentor(img)))
        else:
            self.augmentor = None

        # Replace these with Standard Numpy transforms rather than using PIL images
        # func.extend([transforms.ToPILImage(), transforms.CenterCrop(112), transforms.Grayscale()])
        func.append(transforms.Lambda(lambda img: self._preprocess(img, 112)))
        func.append(transforms.ToTensor())
        self.transforms = transforms.Compose(func)

        assert isinstance(config.data_path_a, str), "Invalid data_path_a, expected a string"
        assert isinstance(config.data_path_b, str), "Invalid data_path_b, expected a string"

        self.dataset_a = self._load_file_list(config.data_path_a)
        self.dataset_b = self._load_file_list(config.data_path_b)

        self.label_filter = config.label_filter if 'label_filter' in config and len(config.label_filter) > 0 else None

        assert len(self.dataset_a) == len(self.dataset_b), "Error: datasets do not match in length"

    def _load_file_list(self, path):
        img_list = []
        for line in open(path, 'r'):
            img_list.append(line.strip())
        return(img_list)

    def _preprocess(self, x, crop=None):
        x = toGrayscale(x)
        if crop:
            x = cropCenter(x, (crop, crop))

        # x = (x - np.min(x)) / (np.max(x) - np.min(x))
        # x -= np.mean(x)
        x = (x - np.mean(x))/np.std(x)
        return(x)

    def _load_and_label(self, index):
        img_a = imread(self.dataset_a[index])
        img_b = imread(self.dataset_b[index])

        if len(img_a.shape) < 3:
            img_a = np.expand_dims(img_a, axis=2)

        if len(img_b.shape) < 3:
            img_b = np.expand_dims(img_b, axis=2)

        name_a = os.path.basename(self.dataset_a[index])
        name_b = os.path.basename(self.dataset_b[index])

        y = np.zeros((2), dtype=np.float32)

        # Override when we have negative examples in a seperate folder
        # Remeber to override this for the validation dataset where only file names count
        if self.label_filter:
            if (self.label_filter in name_a or self.label_filter in name_b):
                y[0] = 1
            else:
                idx = int(get_idx(name_a) == get_idx(name_b))
                y[idx] = 1
        else:
            idx = int(get_idx(name_a) == get_idx(name_b))
            y[idx] = 1

        return img_a, img_b, y

    def __getitem__(self, index):
        # Fix the random state so we get the same transformations
        if self.augmentor:
            self.augmentor.refresh_random_state()

        img_a, img_b, y = self._load_and_label(index)

        a, b = img_a, img_b
        img_a = self.transforms(img_a)
        img_b = self.transforms(img_b)

        # plot_side_by_side(imgs=[a, img_a.numpy(), b, img_b.numpy()])

        return ((img_a, img_b), y)

    def __len__(self):
        return len(self.dataset_a)

    def get_validation_split(self, config_val):
        return NotImplementedError
