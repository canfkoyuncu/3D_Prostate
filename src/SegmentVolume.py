import cv2
import os

import matplotlib
import numpy as np


from src.OTLS_Options import OTLS_Options
from src.Utils import bw_on_image, find_edge
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, Axes3D

from src.VolumeProcessor import eliminateObjectsOnBackground, load_volume_from_h5, load_volume_from_np

matplotlib.use("Qt5Agg")

opt = OTLS_Options().parse()


def main():
    ''' ----------- load data --------------- '''
    datapath = os.path.join(opt.data_path, opt.sample_name, "data-f0.h5")
    prem_nuclei_path = os.path.join(opt.prem_data_path, opt.sample_name + '_nuclei_' + opt.downsample_level)
    prem_cyto_path = os.path.join(opt.prem_data_path, opt.sample_name + '_cyto_' + opt.downsample_level)

    nuclei_vol = load_volume_from_h5(datapath, opt.downsample_level, True, prem_nuclei_path)
    print("Nuclei data has been loaded.")
    nuclei_vol.normalize()

    nuclei_bw = load_volume_from_h5(datapath, opt.downsample_level, True, prem_nuclei_path)
    nuclei_bw.binarize(1.2)

    tissue_vol = load_volume_from_h5(datapath, opt.downsample_level, False, prem_cyto_path)
    print("Tissue data has been loaded.")

    tissue_mask = tissue_vol.binarize(inplace=False)
    tissue_mask.get_convexhull()

    tissue_slic = tissue_vol.normalize(False)
    tissue_slic.slic(100000, 10, True)
    tissue_slic.save_volume(prem_cyto_path+'slic100000_10')
    #tissue_slic = load_volume_from_np(prem_cyto_path+'slic100000_10')
    print("Tissue slic data has been loaded.")

    glandular_region = tissue_vol.binarize(th=1., bgFlag=True, inplace=False)
    print("Gland data has been loaded.")
    eliminateObjectsOnBackground(tissue_slic, tissue_mask)
    print('Non tissue regions eliminated')
    eliminateObjectsOnBackground(tissue_slic, glandular_region)
    print('Cyto tissue regions eliminated')
    tissue_slic.save_volume(prem_cyto_path+'slic100000_10_processed')
    tissue_slic = load_volume_from_np(prem_cyto_path+'slic100000_10_processed')

    '''nuclei_bw.remove_small_objects(100)
    mind, maxd=0, 2500
    minr, maxr=50, 150
    minc, maxc=50, 150

    tissue_slic.visualize(minr=minr, maxr=maxr, minc=minc, maxc=maxc, mind=mind, maxd=maxd, showFlag=True)
    #nuclei_bw.visualize(minr=minr, maxr=maxr, minc=minc, maxc=maxc, mind=mind, maxd=maxd, showFlag=True, color=(0,1,0),transparent=True)
    return'''

    segments = tissue_slic.get_volume()
    print(np.max(segments))
    tissue_vol.normalize()

    #black = np.zeros((nuclei_bw.get_height(), nuclei_bw.get_width()), dtype=np.uint8)
    out = cv2.VideoWriter('../output/' + opt.sample_name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps=5, frameSize=(nuclei_vol.get_width()*2, nuclei_vol.get_height()))
    for i in range(0, nuclei_vol.get_depth(), 2):
        cyt = tissue_vol.get_slice(i)
        cyt = np.dstack((cyt, cyt, cyt)).astype(np.uint8)

        img = nuclei_vol.get_slice(i)
        img = np.dstack((img, img, img)).astype(np.uint8)

        bw = nuclei_bw.get_slice(i)
        bw2 = tissue_mask.get_slice(i)
        #bw3 = gland_bw.get_slice(i)

        bw = find_edge(bw, strel=3)
        bw2 = find_edge(bw2, strel=3)
        #bw3 = find_edge(bw3, strel=3)

        bw_on_image(img, bw, (0, 255, 255))
        bw_on_image(img, bw2, (255, 0, 0))
        #bw_on_image(img, bw3, (255, 255, 0))

        #img = cv2.putText(img=img, text=f"{i}", org=(10, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0), thickness=1)

        for r in range(1, tissue_slic.get_height()-1):
            for c in range(1, tissue_slic.get_width()-1):
                l = segments[r, c, i]
                if l != segments[r,c-1,i] or l != segments[r,c+1,i] or l != segments[r-1,c,i] or l != segments[r+1,c,i]:
                    img[r,c,0] = 0
                    img[r,c,1] = 255
                    img[r,c,2] = 0

        img = np.hstack((img, cyt))
        '''print(i)
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(cyt)
        plt.show()'''
        out.write(img)
    out.release()


if __name__ == "__main__":
    main()
