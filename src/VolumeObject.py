import os

import matplotlib.pyplot as plt
import cv2
import h5py
import scipy
from mayavi import mlab
from scipy import ndimage
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial.qhull import Delaunay
from skimage import segmentation
from skimage.measure import marching_cubes_lewiner
from skimage.morphology import local_maxima, convex_hull_image, convex_hull_object

from src.Utils import threshold, eliminate_small_area, convexhull_volume, eliminate_small_holes


class Volume:
    def __init__(self, volume):
        self.__volume = volume
        self.__height = self.__volume.shape[0]
        self.__width = self.__volume.shape[1]
        self.__depth = self.__volume.shape[2]
        self.print_info()

    def print_info(self):
        print(f'{self.__width}, {self.__height}, {self.__depth}')
        print(np.max(self.__volume), np.min(self.__volume), self.__volume.dtype)

    def check_slice_index(self, index):
        if 0 <= index < self.__depth:
            return True
        else:
            #print('invalid slice index.')
            return False

    def get_depth(self):
        return self.__depth

    def get_height(self):
        return self.__height

    def get_width(self):
        return self.__width

    def get_slice(self, index):
        if index < 0 or index >= self.get_depth():
            return None
        else:
            img = self.__volume[:,:,index]
        return img

    def get_slices(self, indices):
        img = self.__volume[:,:,indices]
        return img

    def set_slice(self, index, slice):
        if self.get_depth() > index >= 0:
            self.__volume[:,:,index] = slice

    def show_slice(self, index):
        slice = self.get_slice(index)
        plt.imshow(slice)
        plt.show()

    #This function creates an image frame for video recording
    def slice_2_frame(self, slice_index):
        img = self.get_slice(slice_index).astype(np.uint8)
        img = cv2.putText(img=img, text=f"{slice_index}", org=(25, 25), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255,0,0), thickness=2)
        return img

    def set_volume(self, volume):
        self.__volume = volume
        self.__height = self.__volume.shape[0]
        self.__width = self.__volume.shape[1]
        self.__depth = self.__volume.shape[2]

    def save_video(self, filename):
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), fps=5, frameSize=(self.__width, self.__height))
        offset = 5
        for i in range(100, self.get_depth(), offset):
        #for i in range(200, 5000, 50):
            frame = self.slice_2_frame(i)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            '''if i%500 == 0:
                print(i)
                plt.imshow(frame)
                plt.show()'''
            out.write(frame)
        out.release()

    def get_volume(self, minr=-1, maxr=-1, minc=-1, maxc=-1, mind=-1, maxd=-1):
        if minr < 0 or maxr >= self.get_height() or minc < 0 or maxc >= self.get_width() or mind < 0 or maxd >= self.get_depth():
            return self.__volume
        else:
            return self.__volume[minr:maxr, minc:maxc, mind:maxd]

    def get_voxel(self, i, j, k):
        return self.__volume[i][j][k]

    def set_voxel(self, i, j, k, val):
        self.__volume[i][j][k] = val

    def binarize(self, th=-1, threshold_func='Otsu', bgFlag=False, inplace=True):
        if inplace:
            self.__volume = threshold(self.__volume, th, bgFlag, threshold_func=threshold_func)
            self.__volume = self.__volume.astype(np.bool)
        else:
            volume = Volume(self.__volume.copy())
            volume.binarize(th, threshold_func=threshold_func, bgFlag=bgFlag, inplace=True)
            return volume

    def binarizeSliceBySlice(self, th=-1, threshold_func='Otsu', bgFlag=False, inplace=True):
        if inplace:
            volume = self.__volume.astype(np.bool)
            for i in range(0, self.__depth):
                volume[:,:,i] = threshold(self.__volume[:,:,i], th, bgFlag, threshold_func=threshold_func)
            self.__volume = volume
        else:
            volume = Volume(self.__volume.copy())
            volume.binarizeSliceBySlice(th, threshold_func=threshold_func, bgFlag=bgFlag, inplace=True)
            return volume

    def remove_small_objects(self, volTh, inplace=True):
        if inplace:
            eliminate_small_area(self.__volume, volTh, in_place=True)
        else:
            volume = Volume(self.__volume.copy())
            volume.remove_small_objects(volTh, True)
            return volume

    def remove_small_holes(self, inplace=True):
        if inplace:
            eliminate_small_holes(self.__volume, in_place=True)
        else:
            volume = Volume(self.__volume.copy())
            volume.remove_small_holes(inplace=True)
            return volume

    def resize(self, scales, inplace=True):
        if inplace:
            self.__volume = ndimage.zoom(self.__volume, scales)
            self.__height = self.__volume.shape[0]
            self.__width = self.__volume.shape[1]
            self.__depth = self.__volume.shape[2]
        else:
            volume = Volume(self.__volume.copy())
            volume.resize(scales, True)
            return volume

    def gaussian_filter(self, sigma, inplace=True):
        if inplace:
            ndimage.filters.gaussian_filter(self.__volume, sigma, output=self.__volume)
        else:
            volume = Volume(self.__volume.copy())
            volume.gaussian_filter(sigma, True)
            return volume

    def normalize(self, inplace=True):
        if inplace:
            #info = np.finfo(self.__volume.dtype) # Get the information of the incoming image type
            self.__volume /= np.max(self.__volume)
            self.__volume *= 255
            #self.__volume /= 20
            #self.__volume[self.__volume>255] = 255
        else:
            volume = Volume(self.__volume.copy())
            volume.normalize(True)
            return volume

    def get_convexhull(self, inplace=True):
        if inplace:
            #hull = convexhull_volume(self.__volume)
            #idx = np.stack(np.indices(self.__volume.shape), axis = -1)
            #deln = Delaunay(hull.points[hull.vertices])
            #out_idx = np.nonzero(deln.find_simplex(idx) + 1)
            #self.__volume[out_idx] = 1
            for i in range(0, self.get_depth()):
                slice = self.get_slice(i)
                self.set_slice(i, convex_hull_image(slice))
        else:
            volume = self.copy()
            volume.get_convexhull(True)
            return volume

    def visualize(self, minr=-1, maxr=-1, minc=-1, maxc=-1, mind=-1, maxd=-1, color=(1,0,0), spacing=(1,1,1), showFlag=True, transparent=False):
        if minr == -1:
            minr, maxr = 0, self.__height-1
        if minc == -1:
            minc, maxc = 0, self.__width-1
        if mind == -1:
            mind, maxd = 0, self.__depth-1

        verts, faces, normals, values = marching_cubes_lewiner(self.__volume[minr:maxr, minc:maxc, mind:maxd], spacing=spacing)
        mesh = mlab.triangular_mesh([vert[0] for vert in verts], [vert[1] for vert in verts], [vert[2] for vert in verts], faces, color=color, transparent=transparent)
        if showFlag:
            mlab.show()
        return verts, faces, normals, values, mesh

    def label(self, inplace=True):
        if inplace:
            self.__volume, n = label(self.__volume)
            self.set_type(np.int)
            print("Labeled image: ", n, self.__volume.dtype)
        else:
            volume = Volume(self.__volume.copy())
            volume.label(True)
            return volume

    def slic(self, n_segments, compactness, inplace=True):
        if inplace:
            self.__volume = segmentation.slic(self.__volume, n_segments=n_segments, compactness=compactness, multichannel=False)
            print(self.__volume.shape)
        else:
            volume = Volume(self.__volume.copy())
            volume.slic(n_segments, compactness, True)
            return volume

    def max_label(self):
        return np.max(self.__volume)

    def save_volume(self, outname):
        np.save(outname + '.npy', self.__volume)

    def set_type(self, type):
        self.__volume = self.__volume.astype(type)

    def morph_closing(self, structure, iterations=1, inplace=True):
        if inplace:
            scipy.ndimage.binary_closing(self.__volume, structure=structure, iterations=iterations, output=self.__volume)
        else:
            volume = Volume(self.__volume.copy())
            volume.morph_closing(structure, iterations, True)
            return volume

    def median_filter(self, size=None, inplace=True):
        if inplace:
            ndimage.filters.median_filter(self.__volume, size=size, output=self.__volume)
        else:
            volume = Volume(self.__volume)
            volume.median_filter(size, True)
            return volume

    def remove_thin_objects(self, th):
        max_label = self.max_label()
        start_end_h = np.ones((max_label+1, 2))*-1
        start_end_w = np.ones((max_label+1, 2))*-1
        start_end_d = np.ones((max_label+1, 2))*-1
        for i in range(0, self.get_height()):
            for j in range(0, self.get_width()):
                for k in range(0, self.get_depth()):
                    label = self.get_voxel(i,j,k)
                    if label > 0:
                        if start_end_h[label, 0] == -1 or start_end_h[label, 0] > i:
                            start_end_h[label, 0] = i
                        if start_end_h[label, 1] == -1 or start_end_h[label, 1] < i:
                            start_end_h[label, 1] = i

                        if start_end_w[label, 0] == -1 or start_end_w[label, 0] > j:
                            start_end_w[label, 0] = j
                        if start_end_w[label, 1] == -1 or start_end_w[label, 1] < j:
                            start_end_w[label, 1] = j

                        if start_end_d[label, 0] == -1 or start_end_d[label, 0] > k:
                            start_end_d[label, 0] = k
                        if start_end_d[label, 1] == -1 or start_end_d[label, 1] < k:
                            start_end_d[label, 1] = k

        elim_indices = np.zeros((max_label+1, 1))
        for i in range(1, max_label+1):
            if start_end_w[i, 0] != -1 and (start_end_w[i, 1] - start_end_w[i, 0]) < th:
                elim_indices[i] = 1
            elif start_end_h[i, 0] != -1 and (start_end_h[i, 1] - start_end_h[i, 0]) < th:
                elim_indices[i] = 1
            elif start_end_d[i, 0] != -1 and (start_end_d[i, 1] - start_end_d[i, 0]) < th:
                elim_indices[i] = 1

        for i in range(0, self.get_height()):
            for j in range(0, self.get_width()):
                for k in range(0, self.get_depth()):
                    label = self.get_voxel(i,j,k)
                    if label > 0 and elim_indices[label] == 1:
                        self.set_voxel(i,j,k,0)

    def relabel(self):
        self.binarize(0)
        self.label()

    def copy(self):
        return Volume(self.__volume.copy())
