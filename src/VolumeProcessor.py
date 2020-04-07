import numpy as np
import os
import h5py
from scipy.ndimage import labeled_comprehension
from skimage.measure import regionprops

from src.VolumeObject import Volume


def eliminateObjectsOnBackground(objects, tissue):
    step = 2
    onTissueVoxels = np.zeros((objects.max_label()+1, 1), dtype=np.float)
    volumes = np.zeros_like(onTissueVoxels)
    for i in range(0, objects.get_height(), step):
        for j in range(0, objects.get_width(), step):
            for k in range(0, objects.get_depth(), step):
                if tissue.get_voxel(i,j,k) > 0:
                    onTissueVoxels[objects.get_voxel(i,j,k)] += 1
                volumes[objects.get_voxel(i,j,k)] += 1

    '''objects_v = objects.get_volume()
    tissue_v = tissue.get_volume()
    labels = range(0, objects.max_label()+1)
    onTissueVoxels = labeled_comprehension(tissue_v, objects_v, labels, np.sum, float, 0)
    volumes = labeled_comprehension(objects_v, objects_v, labels, np.sum, float, 0)'''
    print(len(onTissueVoxels), len(volumes))

    '''onTissueVoxels = np.divide(onTissueVoxels, volumes)
    elimIndices = [i for i, val in enumerate(onTissueVoxels) if val < 0.5]

    for i in elimIndices:
        objects_v[objects_v==i] = 0'''

    for i in range(0, objects.get_height()):
        for j in range(0, objects.get_width()):
            for k in range(0, objects.get_depth()):
                label = objects.get_voxel(i,j,k)
                if label > 0 and onTissueVoxels[label] / volumes[label] < 0.5:
                    objects.set_voxel(i,j,k,0.)





def load_volume_from_h5(filename, downsample_level, isNuclei, outname):
        if os.path.exists(outname + '.npy'):
            print("File exists. Loading..")
            volume = np.load(outname + '.npy')
        else:
            file = h5py.File(filename, 'r')
            depth = len(file['t00000/s00/' + downsample_level + '/cells'])
            height = file['t00000/s00/' + downsample_level + '/cells'][0].shape[0]
            width = file['t00000/s00/' + downsample_level + '/cells'][0].shape[1]
            volume = np.zeros((height, width, depth))
            for i in range(0, depth):
                if isNuclei:
                    volume[:,:,i] = file['t00000/s00/' + downsample_level + '/cells'][i]
                else:
                    volume[:,:,i] = file['t00000/s01/' + downsample_level + '/cells'][i]

                if i % 100 == 0:
                    print(i)
                if i % 500 == 0:
                    np.save(outname + '.npy', volume)

            np.save(outname + '.npy', volume)

        return Volume(volume)


def load_volume_from_np(filename):
    return Volume(np.load(filename+'.npy'))
