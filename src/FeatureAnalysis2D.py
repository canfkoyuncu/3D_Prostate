import os
import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops

from src.OTLS_Options import OTLS_Options
from src.VolumeProcessor import load_volume_from_np

import matplotlib.pyplot as plt


def extractAreaFt4OneSample(datapath):
    numberOfSlices = 10
    tissue_slic = load_volume_from_np(datapath)
    randomIndices = np.random.randint(500, high=1500, size=(numberOfSlices, 1))
    areas = []
    for ind in randomIndices:
        slice = tissue_slic.get_slice(ind)>0
        slice, n = label(slice)
        for region in regionprops(slice):
            if region.area > 50:
                areas.append(region.area)

    return areas


def extractAreaFts(sample_names):
    fts = []
    for sample_name in sample_names:
        print(sample_name)
        datapath = os.path.join(opt.output_path, sample_name + '_cyto_' + opt.downsample_level + 'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+'_processed')
        areas = extractAreaFt4OneSample(datapath)
        fts.extend(areas)

    return fts


if __name__ == '__main__':
    opt = OTLS_Options().parse()
    sample_names = opt.sample_names.split(',')
    fts = extractAreaFts(sample_names)
    print(fts, len(fts))
    ones = []

    for i in range(len(fts)):
        ones.append(1)

    plt.scatter(ones,fts)
    plt.show()

