import multiprocessing
from multiprocessing import Pool
import cv2
import os

import matplotlib
import numpy as np


from src.OTLS_Options import OTLS_Options
from src.Utils import bw_on_image, find_edge
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, Axes3D

from src.VolumeProcessor import eliminateObjectsOnBackground, load_volume_from_h5, load_volume_from_np, \
    save_volume_rendering_as_gif


#matplotlib.use("Qt5Agg")

def main(sample_name):
    ''' ----------- load data --------------- '''
    datapath = os.path.join(opt.data_path, sample_name, "data-f0.h5")
    prem_nuclei_path = os.path.join(opt.prem_data_path, sample_name + '_nuclei_' + opt.downsample_level)
    prem_cyto_path = os.path.join(opt.prem_data_path, sample_name + '_cyto_' + opt.downsample_level)

    nuclei_vol = load_volume_from_h5(datapath, opt.downsample_level, True, prem_nuclei_path)
    print("Nuclei data has been loaded.")

    nuclei_bw = nuclei_vol.binarize(1.2, inplace=False)
    print("Nuclei mask has been loaded.")

    tissue_vol = load_volume_from_h5(datapath, opt.downsample_level, False, prem_cyto_path)
    print("Tissue data has been loaded.")

    tissue_mask = tissue_vol.binarize(inplace=False)
    tissue_mask.get_convexhull()
    print("Tissue mask has been loaded.")

    tissue_slic = tissue_vol.normalize(inplace=False)
    tissue_slic.slic(opt.slic_no, opt.slic_compactness)
    tissue_slic.save_volume(prem_cyto_path+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness))
    #tissue_slic = load_volume_from_np(prem_cyto_path+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness))
    print("Tissue slic data has been loaded.")

    glandular_region = tissue_vol.binarize(bgFlag=True, inplace=False)
    print("Gland data has been loaded.")
    eliminateObjectsOnBackground(tissue_slic, tissue_mask)
    print('Non tissue regions eliminated')
    eliminateObjectsOnBackground(tissue_slic, glandular_region)
    print('Cyto tissue regions eliminated')
    tissue_slic.save_volume(os.path.join(opt.output_path, sample_name + '_cyto_' + opt.downsample_level+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+'_processed'))
    #tissue_slic = load_volume_from_np(os.path.join(opt.output_path, sample_name + '_cyto_' + opt.downsample_level+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+'_processed'))

    '''nuclei_bw.remove_small_objects(100)
    mind, maxd=1000, 1500
    minr, maxr=50, 150
    minc, maxc=50, 150

    tissue_slic.visualize(minr=minr, maxr=maxr, minc=minc, maxc=maxc, mind=mind, maxd=maxd, showFlag=False)
    nuclei_bw.visualize(minr=minr, maxr=maxr, minc=minc, maxc=maxc, mind=mind, maxd=maxd, showFlag=False, color=(0,1,0),transparent=True)
    video_path = os.path.join(opt.output_path, sample_name)
    #os.makedirs(video_path)
    save_volume_rendering_as_gif(video_path, sample_name)
    return'''

    nuclei_vol.normalize()
    print(tissue_slic.max_label())
    tissue_vol.normalize()
    out = cv2.VideoWriter(os.path.join(opt.output_path, sample_name+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+ '.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), fps=5, frameSize=(nuclei_vol.get_width()*3, nuclei_vol.get_height()))
    for i in range(0, nuclei_vol.get_depth(), 2):
        cyt = tissue_vol.get_slice(i)
        cyt3 = np.dstack((cyt, cyt, cyt)).astype(np.uint8)

        nuclei = nuclei_vol.get_slice(i)
        nuclei3 = np.dstack((nuclei, nuclei, nuclei)).astype(np.uint8)
        res = np.dstack((nuclei, cyt, nuclei)).astype(np.uint8)

        bw = nuclei_bw.get_slice(i)
        bw2 = tissue_mask.get_slice(i)
        #bw3 = gland_bw.get_slice(i)

        bw = find_edge(bw, strel=3)
        bw2 = find_edge(bw2, strel=3)
        #bw3 = find_edge(bw3, strel=3)

        bw_on_image(res, bw, (0, 255, 255))
        bw_on_image(res, bw2, (255, 0, 0))
        #bw_on_image(res, bw3, (255, 255, 0))

        #img = cv2.putText(img=img, text=f"{i}", org=(10, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0), thickness=1)

        for r in range(1, tissue_slic.get_height()-1):
            for c in range(1, tissue_slic.get_width()-1):
                l = tissue_slic.get_voxel(r, c, i)
                if l != tissue_slic.get_voxel(r,c-1,i) or l != tissue_slic.get_voxel(r,c+1,i) or l != tissue_slic.get_voxel(r-1,c,i) or l != tissue_slic.get_voxel(r+1,c,i):
                    res[r,c,0] = 0
                    res[r,c,1] = 0
                    res[r,c,2] = 255

        img = np.hstack((cyt3, nuclei3, res))
        '''print(i)
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
        ax[0].imshow(img)
        ax[1].imshow(cyt)
        plt.show()'''
        out.write(img)
    out.release()


if __name__ == "__main__":
    opt = OTLS_Options().parse()
    sample_names = opt.sample_names.split(',')
    main(sample_names[0])
    #pool = Pool()
    #pool.map(main, sample_names)
    #pool.close()
    #pool.join()

