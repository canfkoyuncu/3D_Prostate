import numpy as np
import os
import h5py
import subprocess
from mayavi import mlab
from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes

neighbors2d = [[0, -1], [-1, 0], [-1, -1], [0, 1], [1, 0], [1, 1], [1, -1], [-1, 1]]


neighbors3d = [[0, -1, 0], [-1, 0, 0], [0, 0, -1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1], [-1, -1, -1],
               [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
               [1, -1, 0], [1, -1, 1], [1, 1, -1], [1, 0, -1], [1, -1, -1], [0, -1, 1], [0, 1, -1],
               [-1, 1, 0], [-1, 1, 1], [-1, 0, 1], [-1, -1, 1], [-1, 1, -1], [0, 0, 0]]


from src.VolumeObject import Volume


def get_volume_fts(objects):
    step = 1
    fts = np.zeros((objects.max_label()+1, 1), dtype=np.float)
    for i in range(0, objects.get_height(), step):
        for j in range(0, objects.get_width(), step):
            for k in range(0, objects.get_depth(), step):
                label = objects.get_voxel(i,j,k)
                if label > 0:
                    fts[label] += 1

    fts = np.array(fts)
    return fts[np.nonzero(fts)]


def eliminate_objects_in_background(objects, tissue):
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


def save_volume_rendering_as_gif(outpath, sample_name, fps=20):
    padding = 5
    '''@mlab.animate()
    def anim():
        flag = 10
        for i in range(1,720,1):
            print("view: {}\nroll: {}".format(mlab.view(), mlab.roll()))
            if i%50 == 0:
                flag *= -1
            mlab.move(forward=flag, right=1, up=flag)
            mlab.view(i, elevation=i)
            mlab.show()
            # create zeros for padding index positions for organization
            zeros = '0'*(padding - len(str(i)))

            # concate filename with zero padded index number as suffix
            filename = os.path.join(outpath, '{}_{}{}{}'.format(sample_name, zeros, i, '.png'))

            mlab.savefig(filename=filename)

            yield

    mlab.view(1, elevation=1)
    f = mlab.gcf()
    f.scene.movie_maker.record = True
    anim()
    mlab.show()'''

    ffmpeg_fname = os.path.join(outpath, '{}_%0{}d{}'.format(sample_name, padding, '.png'))
    cmd = 'ffmpeg -f image2 -r {} -i {} -vcodec gif -y {}.gif'.format(fps,
                                                                        ffmpeg_fname,
                                                                        sample_name)
    print(cmd)
    subprocess.check_output(['bash','-c', cmd])


import matplotlib.pyplot as plt
def eliminate_objects_cell_surrounding(objects, cells, th=0.50):
    dist_th = 5
    if objects.max_label() == 0:
        return
    for k in range(0, objects.get_depth()):
        sl_object = objects.get_slice(k) #assume objects has binary volume
        sl_object, n = ndimage.label(sl_object)
        if n == 0:
            continue
        sl_cells = cells.get_slice(k)
        dist = ndimage.morphology.distance_transform_edt(sl_cells==0)
        objectDists = np.zeros((n+1, 1), dtype=np.float)
        cnts = np.zeros((n+1, 1), dtype=np.float)
        for i in range(1, objects.get_height()-1):
            for j in range(1, objects.get_width()-1):
                label = sl_object[i,j]
                if label > 0:
                    onBorder = False
                    for d in neighbors2d:
                        i2 = i + d[0]
                        j2 = j + d[1]
                        if sl_object[i2,j2] != label:
                            onBorder = True
                            break

                    if onBorder:
                        if dist[i,j] < dist_th:
                            objectDists[label] += 1
                        cnts[label] += 1

        for i, j in enumerate(objectDists):
            if cnts[i] > 0 and j / cnts[i] < th:
                sl_object[sl_object==i] = 0

        objects.set_slice(k, sl_object)

def eliminate_objects_cell_surrounding2(objects, cells, th=0.25):
    dist_th = 5
    offset=2
    cells_v = cells.get_volume()==0
    objectDists = np.zeros((objects.max_label()+1, 1), dtype=np.float)
    cnts = np.zeros((objects.max_label()+1, 1), dtype=np.float)
    dist = ndimage.morphology.distance_transform_edt(cells_v)
    for i in range(1, objects.get_height()-1, offset):
        for j in range(1, objects.get_width()-1, offset):
            for k in range(1, objects.get_depth()-1, offset):
                label = objects.get_voxel(i,j,k)
                if label > 0:
                    onBorder = False
                    minDist = np.Inf
                    for d in neighbors3d:
                        i2 = i + d[0]
                        j2 = j + d[1]
                        k2 = k + d[2]
                        if objects.get_voxel(i2,j2,k2) != label:
                            onBorder = True
                            if minDist > dist[i,j,k]:
                                minDist = dist[i,j,k]

                    if onBorder:
                        if minDist < dist_th:
                            objectDists[label] += 1
                        cnts[label] += 1

    for i in range(1, objects.get_height()-1):
        for j in range(1, objects.get_width()-1):
            for k in range(1, objects.get_depth()-1):
                label = objects.get_voxel(i,j,k)
                if label > 0 and objectDists[label] / cnts[label] < th:
                     objects.set_voxel(i,j,k, 0)


def regions_surrounded_cells(objects, cells):
    if cells.max_label() == 0:
        return
    for k in range(0, cells.get_depth()):
        sl_cells = cells.get_slice(k)
        sl_cells = np.bitwise_xor(sl_cells, remove_small_holes(sl_cells, 10000000))

        sl_objects = objects.get_slice(k)
        sl_objects[sl_cells] = 1
        objects.set_slice(k, sl_objects)
