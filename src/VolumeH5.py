import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.measure import marching_cubes_lewiner
import cv2
import h5py

from src.Utils import get_arguments, bw_on_image, find_edge

neighbors3d = [[0, -1, 0], [-1, 0, 0], [0, 0, -1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1], [-1, -1, -1],
               [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
               [1, -1, 0], [1, -1, 1], [1, 1, -1], [1, 0, -1], [1, -1, -1], [0, -1, 1], [0, 1, -1],
               [-1, 1, 0], [-1, 1, 1], [-1, 0, 1], [-1, -1, 1], [-1, 1, -1], [0, 0, 0]]

neighbors2d = [[0, -1], [-1, 0], [-1, -1], [0, 1], [1, 0], [1, 1], [1, -1], [-1, 1]]

args = get_arguments()


class Volume:
    def __init__(self):
        try:
            self.file = h5py.File(args["data_path"], 'r')

            self.__depth = len(self.file['t00000/s00/' + args['downsample_level'] + '/cells'])
            self.__height = self.file['t00000/s00/' + args['downsample_level'] + '/cells'][0].shape[0]
            self.__width = self.file['t00000/s00/' + args['downsample_level'] + '/cells'][0].shape[1]
            self.__channel = 2
            self.__step_size = 1
        except FileNotFoundError:
            print ("File couldnt be loaded..")

    def print_info(self):
        print(f'{self.__width}, {self.__height}, {self.__channel}, {self.__depth}')

    def get_slice(self, index, nucleiFlag=True):
        if index < 0 or index >= self.get_depth():
            return None
        elif nucleiFlag: #s00 stores nuclei, s01 stores cytoplasm, normalized the voxels by 25. but should be checked later
            img = self.file['t00000/s00/' + args['downsample_level'] + '/cells'][index]/25.
        else:
            img = self.file['t00000/s01/' + args['downsample_level'] + '/cells'][index]/25.
        return img

    def check_slice_index(self, index):
        if 0 <= index < self.__depth:
            return True
        else:
            #print('invalid slice index.')
            return False

    def check_channel_index(self, index):
        if 0 <= index < self.__channel:
            return True
        else:
            #print('invalid channel index.')
            return False

    def get_depth(self):
        return self.__depth

    def get_height(self):
        return self.__height

    def get_width(self):
        return self.__width

    def get_channel(self):
        return self.__channel

    #This function creates an image frame for video recording
    def slice_2_frame(self, slice_index, nucleiFlag):
        img = self.get_slice(slice_index, nucleiFlag).astype(np.uint8)
        img = cv2.putText(img=img, text=f"{slice_index}", org=(25, 25), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255,0,0), thickness=2)
        return img

    def save_video(self, filename, nucleiFlag):
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), fps=5, frameSize=(self.__width, self.__height))
        offset = 5
        #for i in range(0, self.get_depth(), offset):
        for i in range(200, 5000, 50):
            frame = self.slice_2_frame(i, nucleiFlag)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            '''if i%500 == 0:
                print(i)
                plt.imshow(frame)
                plt.show()'''
            out.write(frame)
        out.release()
