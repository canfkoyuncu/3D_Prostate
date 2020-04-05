import h5py

import cv2
import os

import matplotlib
import numpy as np
from mayavi import mlab
from scipy.spatial.qhull import Delaunay
from skimage.measure import marching_cubes_lewiner

from src.OTLS_Options import OTLS_Options
from src.Utils import bw_on_image, find_edge
from src.VolumeObject import Volume
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, Axes3D


import napari

matplotlib.use("Qt5Agg")

opt = OTLS_Options().parse()


def main():
    ''' ----------- load data --------------- '''
    datapath = os.path.join(opt.data_path, opt.sample_name, "data-f0.h5")
    prem_nuclei_path = os.path.join(opt.prem_data_path, opt.sample_name + '_nuclei_' + opt.downsample_level)
    prem_cyto_path = os.path.join(opt.prem_data_path, opt.sample_name + '_cyto_' + opt.downsample_level)


    # read the data
    # you can download the example data from here:
    # https://hcicloud.iwr.uni-heidelberg.de/index.php/s/6LuE7nxBN3EFRtL
    with h5py.File(datapath, 'r') as f:
        # load the raw data
        print(f['__DATA_TYPES__/'][:])
        print(f['s00'][:])
        print(f['s01'][:])
        print(f['t00000'][:])
        raw = f['raw'][:]
        # load the affinities, we only need the first 3 channels
        affs = f['affinities'][:3, :]

    # visualize the raw data and the affinity maps with napari
    # TODO switch to new napari syntax
    # napari.view_image(raw, name='raw')
    # napari.view_image(affs, name='affinities')
    viewer = napari.Viewer()
    viewer.add_image(raw, name='raw')
    viewer.add_image(affs, name='affinities')
    return

    nuclei_vol = Volume()
    nuclei_vol.load_volume_from_h5(datapath, opt.downsample_level, True, prem_nuclei_path)
    print("Nuclei data has been loaded.")
    nuclei_vol.normalize()

    tissue_vol = Volume()
    tissue_vol.load_volume_from_h5(datapath, opt.downsample_level, False, prem_cyto_path)
    print("Tissue data has been loaded.")
    tissue_vol.normalize()

    nuclei_bw = Volume()
    nuclei_bw.load_volume_from_h5(datapath, opt.downsample_level, True, prem_nuclei_path)
    nuclei_bw.binarize(1.2)

    tissue_bw = Volume()
    tissue_bw.load_volume_from_h5(datapath, opt.downsample_level, False, prem_cyto_path)
    tissue_bw.binarize()
    tissue_bw.remove_small_objects(1000)
    #tissue_bw.label()
    tissue_bw.set_convexhull()

    gland_bw = Volume()
    gland_bw.load_volume_from_h5(datapath, opt.downsample_level, False, prem_cyto_path)
    print("Gland data has been loaded.")
    gland_bw.binarize(1.2, True)
    gland_bw.remove_small_objects(100)
    gland_bw.watershed(nuclei_bw)

    '''tissue_bw.visualize(0, -1, 0, -1, 0, -1, (0,1,0))
    nuclei_bw.visualize(0, -1, 0, -1, 0, -1, (1,0,0))
    gland_bw.visualize(0, -1, 0, -1, 0, -1, (1,0,1))
    mlab.show()'''


    segments = gland_bw.get_volume()
    print(np.max(segments))

    black = np.zeros((nuclei_bw.get_height(), nuclei_bw.get_width()), dtype=np.uint8)
    #out = cv2.VideoWriter(args['sample_name'] + '_nuclei.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps=5, frameSize=(nuclei_vol.get_width(), nuclei_vol.get_height()))
    for i in range(0, nuclei_vol.get_depth(), 200):
        cyt = tissue_vol.get_slice(i)
        cyt = np.dstack((cyt, cyt, cyt)).astype(np.uint8)

        img = nuclei_vol.get_slice(i)
        img = np.dstack((img, img, img)).astype(np.uint8)

        bw = nuclei_bw.get_slice(i)
        bw2 = tissue_bw.get_slice(i)
        #bw3 = gland_bw.get_slice(i)

        bw = find_edge(bw, strel=3)
        bw2 = find_edge(bw2, strel=3)
        #bw3 = find_edge(bw3, strel=3)

        bw_on_image(img, bw, (0, 255, 0))
        bw_on_image(img, bw2, (255, 0, 0))
        #bw_on_image(img, bw3, (255, 255, 0))

        #img = cv2.putText(img=img, text=f"{i}", org=(10, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0), thickness=1)

        if i%200 == 0:
            for r in range(1, gland_bw.get_height()-1):
                for c in range(1, gland_bw.get_width()-1):
                    l = segments[r, c, i]
                    if l != segments[r,c-1,i] or l != segments[r,c+1,i] or l != segments[r-1,c,i] or l != segments[r+1,c,i]:
                        img[r,c,0] = 0
                        img[r,c,1] = 255
                        img[r,c,2] = 0
            print(i)
            fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
            ax[0].imshow(img)
            ax[1].imshow(cyt)
            plt.show()
        #out.write(img)
    #out.release()


if __name__ == "__main__":
    main()
