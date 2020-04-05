import os

import matplotlib.pyplot as plt
import cv2
import h5py
import scipy
from mayavi import mlab
from scipy import ndimage
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.qhull import Delaunay
from skimage.measure import marching_cubes_lewiner
from skimage.morphology import local_maxima

from src.Utils import threshold, eliminate_small_area, convexhull_volume, label_volume


class Volume:
    def print_info(self):
        print(f'{self.__width}, {self.__height}, {self.__depth}')

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

    def show_slice(self, index):
        slice = self.get_slice(index)
        plt.imshow(slice)
        plt.show()

    #This function creates an image frame for video recording
    def slice_2_frame(self, slice_index, nucleiFlag):
        img = self.get_slice(slice_index).astype(np.uint8)
        img = cv2.putText(img=img, text=f"{slice_index}", org=(25, 25), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255,0,0), thickness=2)
        return img

    def save_video(self, filename, nucleiFlag):
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), fps=5, frameSize=(self.__width, self.__height))
        offset = 5
        for i in range(100, self.get_depth(), offset):
        #for i in range(200, 5000, 50):
            frame = self.slice_2_frame(i, nucleiFlag)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            '''if i%500 == 0:
                print(i)
                plt.imshow(frame)
                plt.show()'''
            out.write(frame)
        out.release()

    def load_volume_from_h5(self, filename, downsample_level, isNuclei, outname):
        if os.path.exists(outname + '.npy'):
            print("File exists. Loading..")
            self.__volume = np.load(outname + '.npy')
            self.__depth = self.__volume.shape[2]
            self.__height = self.__volume.shape[0]
            self.__width = self.__volume.shape[1]
            self.print_info()
            print(np.max(self.__volume), np.min(self.__volume), self.__volume.dtype)
        else:
            file = h5py.File(filename, 'r')
            self.__depth = len(file['t00000/s00/' + downsample_level + '/cells'])
            self.__height = file['t00000/s00/' + downsample_level + '/cells'][0].shape[0]
            self.__width = file['t00000/s00/' + downsample_level + '/cells'][0].shape[1]
            self.__volume = np.zeros((self.__height, self.__width, self.__depth))
            self.print_info()
            for i in range(0, self.__depth):
                if isNuclei:
                    self.__volume[:,:,i] = file['t00000/s00/' + downsample_level + '/cells'][i]
                else:
                    self.__volume[:,:,i] = file['t00000/s01/' + downsample_level + '/cells'][i]

                if i % 100 == 0:
                    print(i)
                if i % 500 == 0:
                    np.save(outname + '.npy', self.__volume)

            np.save(outname + '.npy', self.__volume)

    def get_volume(self, minr=-1, maxr=-1, minc=-1, maxc=-1, mind=-1, maxd=-1):
        if minr < 0 or maxr >= self.get_height() or minc < 0 or maxc >= self.get_width() or mind < 0 or maxd >= self.get_depth():
            return self.__volume
        else:
            return self.__volume[minr:maxr, minc:maxc, mind:maxd]

    def get_voxel(self, i, j, k):
        return self.__volume[i][j][k]

    def binarize(self, th=-1, bgFlag=False):
        self.__volume = threshold(self.__volume, th, bgFlag)

    def remove_small_objects(self, volTh):
        eliminate_small_area(self.__volume, volTh, in_place=True)

    def resize(self, scales):
        self.__volume = ndimage.zoom(self.__volume, scales)
        self.__depth = self.__volume.shape[2]
        self.__height = self.__volume.shape[0]
        self.__width = self.__volume.shape[1]

    def gaussian_filter(self, sigma):
        ndimage.filters.gaussian_filter(self.__volume, sigma, output=self.__volume)

    def normalize(self):
        info = np.finfo(self.__volume.dtype) # Get the information of the incoming image type
        #self.__volume /= np.max(self.__volume)
        #self.__volume *= 255
        self.__volume /= 20
        self.__volume[self.__volume>255] = 255

    def set_convexhull(self):
        hull = convexhull_volume(self.__volume)

        idx = np.stack(np.indices(self.__volume.shape), axis = -1)
        deln = Delaunay(hull.points[hull.vertices])
        out_idx = np.nonzero(deln.find_simplex(idx) + 1)
        self.__volume[out_idx] = 1
        #self.__volume[hull.points[hull.simplices[:], :]] = 1

    def visualize(self, minr, maxr, minc, maxc, mind, maxd, color, spacing=(1,1,1)):
        if maxr == -1:
            maxr = self.__height-1
        if maxc == -1:
            maxc = self.__width-1
        if maxd == -1:
            maxd = self.__depth-1

        verts, faces, normals, values = marching_cubes_lewiner(self.__volume[minr:maxr, minc:maxc, mind:maxd], spacing=spacing)
        mesh = mlab.triangular_mesh([vert[0] for vert in verts], [vert[1] for vert in verts], [vert[2] for vert in verts], faces, color=color)
        return verts, faces, normals, values, mesh

    def label(self):
        self.__volume, label = label_volume(self.__volume)
        print("Labeled image: ", label)

    def watershed(self, cells, mask=None):
        distance = distance_transform_edt(cells.get_volume())
        distance = distance_transform_edt(cells.get_volume())
        markers = local_maxima(distance)
        '''distance = ndimage.maximum_filter(distance, size=10, mode='constant')
        from skimage.feature import peak_local_max
        markers = peak_local_max(distance, min_distance=20, indices=False)
        markers, n = ndimage.label(markers)'''
        print(distance.dtype, self.__volume.dtype, self.__volume.shape)
        self.__volume = scipy.ndimage.measurements.watershed_ift(distance.astype(np.uint8), markers)
        print(self.__volume.shape)
