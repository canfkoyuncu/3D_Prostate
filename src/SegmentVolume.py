import multiprocessing
import skimage
from multiprocessing import Pool
import cv2
import os

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage.filters import try_all_threshold

from src.OTLS_Options import OTLS_Options
from src.Utils import bw_on_image, find_edge, label_on_image, kmeansOTLS_slices, print_label_on_image, \
    calculate_threshold_automatically, threshold, color_deconvolution
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, Axes3D

from src.VolumeProcessor import eliminate_objects_in_background, load_volume_from_h5, load_volume_from_np, \
    save_volume_rendering_as_gif, eliminate_objects_cell_surrounding, get_regions_surrounded_by_cells, remove_background, \
    sum_volumes, add_touching_cells


#matplotlib.use("Qt5Agg")

#tissue_slic.binarize(0)
#subtract_volumes(tissue_slic, nuclei_bw)

def main(sample_name):
    ''' ----------- load data --------------- '''
    datapath = os.path.join(opt.data_path, sample_name, "data-f0.h5")

    nuclei_vol = load_volume_from_h5(datapath, opt.downsample_level, True, os.path.join(opt.output_path, sample_name + '_nuclei_' + opt.downsample_level))
    print("Nuclei data has been loaded.")

    tissue_vol = load_volume_from_h5(datapath, opt.downsample_level, False, os.path.join(opt.output_path, sample_name + '_cyto_' + opt.downsample_level))
    print("Tissue data has been loaded.")
    nuclei_vol.normalize()
    tissue_vol.normalize()

    nuclei_bw = nuclei_vol.binarizeSliceBySlice(threshold_func='Otsu', inplace=False)
    nuclei_bw.label()
    nuclei_bw.remove_small_objects(10000)
    nuclei_bw.binarize(0)

    #tissue_mask = nuclei_bw.get_convexhull(inplace=False)
    #tissue_mask.save_volume(os.path.join(opt.output_path, sample_name + '_' + opt.downsample_level+'_CHull'))
    tissue_mask = load_volume_from_np(os.path.join(opt.output_path, sample_name + '_' + opt.downsample_level+'_CHull'))
    print("Tissue mask size:", tissue_mask.get_volume().shape)
    print("Tissue mask data has been loaded.")

    #tissue_slic = tissue_vol.copy()
    #tissue_slic.slic(opt.slic_no, opt.slic_compactness)
    #tissue_slic.save_volume(os.path.join(opt.output_path, sample_name+'_'+opt.downsample_level+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)))
    #tissue_slic = load_volume_from_np(os.path.join(opt.output_path, sample_name+'_'+opt.downsample_level+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)))
    print("Tissue slic data has been loaded.")

    #glandular_region = tissue_vol.binarizeSliceBySlice(th=1.0, bgFlag=True, threshold_func='Otsu', inplace=False)
    #remove_background(glandular_region, tissue_mask)
    print("Gland data has been loaded.")

    #eliminate_objects_in_background(tissue_slic, glandular_region)
    #tissue_slic.save_volume(os.path.join(opt.output_path, sample_name + '_' + opt.downsample_level+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+'_Glands'))
    tissue_slic = load_volume_from_np(os.path.join(opt.output_path, sample_name + '_' + opt.downsample_level+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+'_Glands'))
    print('Cyto tissue regions eliminated')

    #add_touching_cells(tissue_slic, nuclei_bw)
    #print('Cells are added')

    eliminate_objects_cell_surrounding(tissue_slic, nuclei_bw, th=0.5)
    #tissue_slic.save_volume(os.path.join(opt.output_path, sample_name + '_' + opt.downsample_level+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+'_Glands2'))
    #tissue_slic = load_volume_from_np(os.path.join(opt.output_path, sample_name + '_' + opt.downsample_level+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+'_Glands2'))
    #print('Non-glandular regions eliminated')

    tissue_slic.median_filter((1,1,5))

    '''for i in [100, 250, 500, 645]:#1000, 1250, 2000, 2500]:
        fig, axis = plt.subplots(nrows=2, ncols=2)

        cyt = tissue_vol.get_slice(i)
        nuclei = nuclei_vol.get_slice(i)
        res = np.dstack((nuclei, cyt, nuclei)).astype(np.uint8)
        res2 = res.copy()
        labels = tissue_slic.get_slice(i)
        label_on_image(res2, labels, randSeed=10)

        axis[0,0].imshow(tissue_vol.get_slice(i))
        axis[0,1].imshow(nuclei_vol.get_slice(i))
        axis[1,0].imshow(glandular_region.get_slice(i))
        axis[1,1].imshow(res2)
        plt.show()

    return'''

    #tissue_slic.remove_small_holes()
    #tissue_slic.remove_small_objects(250)
    #tissue_slic.relabel()

    #mind, maxd, minr, maxr, minc, maxc=100, 1500, 50, 150, 50, 150
    #tissue_slic.visualize(minr=minr, maxr=maxr, minc=minc, maxc=maxc, mind=mind, maxd=maxd, showFlag=True)
    #nuclei_bw.visualize(minr=minr, maxr=maxr, minc=minc, maxc=maxc, mind=mind, maxd=maxd, showFlag=True, color=(0,1,0),transparent=True)
    #video_path = os.path.join(opt.output_path, sample_name)
    #os.makedirs(video_path)
    #save_volume_rendering_as_gif(video_path, sample_name)
    #return

    out = cv2.VideoWriter(os.path.join(opt.output_path, sample_name+'slic'+str(opt.slic_no)+'_'+str(opt.slic_compactness)+ '.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), fps=5, frameSize=(nuclei_vol.get_width()*3, nuclei_vol.get_height()))
    for i in range(0, nuclei_vol.get_depth(), 2):
        cyt = tissue_vol.get_slice(i)
        cyt3 = np.dstack((cyt, cyt, cyt)).astype(np.uint8)

        nuclei = nuclei_vol.get_slice(i)
        nuclei3 = np.dstack((nuclei, nuclei, nuclei)).astype(np.uint8)
        res = np.dstack((nuclei, cyt, nuclei)).astype(np.uint8)

        #bw = nuclei_bw.get_slice(i)
        #bw = find_edge(bw, strel=3)
        #bw_on_image(res, bw, (0, 255, 255))

        bw2 = tissue_mask.get_slice(i)
        bw2 = find_edge(bw2, strel=3)
        bw_on_image(res, bw2, (255, 0, 0))

        bw3 = tissue_slic.get_slice(i)>0
        #bw3 = find_edge(bw3, strel=3)
        bw_on_image(res, bw3, (255, 255, 0))
        #label_on_image(res, tissue_slic.get_slice(i), randSeed=0)

        res = cv2.putText(img=res, text=f"{i}", org=(10, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0), thickness=1)

        img = np.hstack((cyt3, nuclei3, res))
        if i % 1000 == -1:
            print(i)
            plt.imshow(img)
            plt.show()
        out.write(img)
    out.release()

import imageio as io
def try_stain_deconv():
    #im = io.imread('/Volumes/data/University of Washington_3D_Prostate_Path/Annotations/18-040_n2_Z000989_annotated.tif')
    #im = im[:, 500:1000]
    #io.imwrite('deneme.png', im)
    im = io.imread('deneme.png')
    im = im[1200:1700,:,:]

    #deconv = color_deconvolution(im.astype(np.float))
    deconv = rgb2hed(im)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im)
    ax[1].imshow(deconv[:, :, 0], cmap='gray')
    ax[2].imshow(deconv[:, :, 1], cmap='gray')
    ax[3].imshow(deconv[:, :, 2], cmap='gray')

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    try_stain_deconv()
    exit(0)
    opt = OTLS_Options().parse()
    sample_names = opt.sample_names.split(',')
    for sample_name in sample_names:
        main(sample_name)

    #pool = Pool()
    #pool.map(main, sample_names)
    #pool.close()
    #pool.join()

