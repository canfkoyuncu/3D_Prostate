import os

from src.OTLS_Options import OTLS_Options
from src.VolumeProcessor import load_volume_from_np, get_volume_fts

import matplotlib.pyplot as plt


def extractVolumeFt4OneSample(datapath):
    tissue_slic = load_volume_from_np(datapath)
    tissue_slic.binarize(1)
    tissue_slic.label()
    tissue_slic.remove_small_objects(250)
    vols = get_volume_fts(tissue_slic)
    return vols


def extractVolumeFts(sample_names):
    fts = []
    for sample_name in sample_names:
        print(sample_name)
        datapath = os.path.join(opt.output_path, sample_name + '_cyto_' + opt.downsample_level + 'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+'_processed')
        vols = extractVolumeFt4OneSample(datapath)
        fts.extend(vols)

    return fts


if __name__ == '__main__':
    opt = OTLS_Options().parse()
    sample_names = opt.sample_names.split(',')
    fts = extractVolumeFts(sample_names)
    print(fts, len(fts))
    ones = []

    for i in range(len(fts)):
        ones.append(1)

    plt.scatter(ones,fts)
    plt.show()

