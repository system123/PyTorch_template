import matplotlib.pyplot as plt
import numpy as np

def plot_side_by_side(imgs=[], n_col=4):
    n_rows = len(imgs)//n_col + 1
    f, axarr = plt.subplots(n_rows, n_col)

    for i, img in enumerate(imgs):
        c = i % n_col
        r = i // n_col

        if len(img.shape) == 3 and img.shape[0] in [1,3,4]:
            img = np.transpose(img, axes=(1,2,0))

        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img[:,:,0]

        axarr[r, c].imshow(img)

    plt.show()
