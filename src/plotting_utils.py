import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

def visualize_array(arr: np.ndarray, aspect: float=1., fp=None, title=None) -> None:
    plt.figure(figsize=(15, 15), facecolor='w')
    min_val = arr.min()
    max_val = arr.max()
    if min_val < 0 and max_val > 0:
        cmap = cm.get_cmap('seismic')
        cmap.set_bad('green')
        
        norm = colors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
        im = plt.imshow(arr, aspect=aspect, norm=norm, cmap=cmap)
    else:
        cmap = cm.get_cmap('hot')
        cmap.set_bad('green')
        im = plt.imshow(arr, aspect=aspect, cmap=cmap)
    plt.colorbar()
    plt.xlabel('features', size=20)
    plt.ylabel('samples', size=20)
    if title is not None:
        plt.title(title, size=20)
        # plt.figure().tight_layout()
    if fp is not None:
        plt.savefig(fp)
    plt.show()
    plt.clf()
