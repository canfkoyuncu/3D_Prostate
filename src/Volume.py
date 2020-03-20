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
    def __init__(self, data_src=None, channel=3):
        try:
            self.__sample_name = f'{args["sample_name"]}_c{args["minc"]}_{args["maxc"]}'
            self.__data = np.load(f'{args["output_path"]}/{self.__sample_name}')
            self.__depth = self.__data.shape[0]
            self.__height = self.__data.shape[1]
            self.__width = self.__data.shape[2]
            self.__channel = self.__data.shape[3]

        except FileNotFoundError:
            print(os.path.exists(args["data_path"]))
            if args["data_path"].endswith(".h5"):
                file = h5py.File(args["data_path"], 'r')
                print(file.keys())
                #for item in file['t00000']['s00']:
                print(file['t00000']['s00']['0']['cells'])

                #print(file['s01']['subdivisions'])
            else:
                print(f"Could not read the file '{data_src}'. Reading images...")
                self.__depth = args["maxd"] - args["mind"]
                self.__height = int((args["maxr"] - args["minr"])*args["ratio"])
                self.__width = int((args["maxc"] - args["minc"])*args["ratio"])
                self.__channel = channel

                self.__data = np.ndarray((self.__depth, self.__height, self.__width, self.__channel), dtype=np.float32)
                for i in range(args["mind"], args["maxd"]):
                    filename = "{}/{}/HE/{}_Z{:06d}.tif".format(args["data_path"], args["sample_name"], args["sample_name"], (i - 1) * 4 + 1)
                    im = Image.open(filename)
                    im = im.crop(box=(args["minc"],args["minr"],args["maxc"],args["maxr"]))
                    im = im.resize((self.__width, self.__height))
                    self.set_slice(i-args["mind"], im)
                    print(filename)
        self.print_info()
        self.__step_size = 1

    def print_info(self):
        print(f'{self.__width}, {self.__height}, {self.__channel}, {self.__depth}, {self.__data.dtype}')

    def get_slice(self, index, channelIndex=-1):
        if self.check_slice_index(index):
            if self.check_channel_index(channelIndex):
                t = self.__data[index, :, :, channelIndex]
            else:
                t = self.__data[index, :, :, :]
            return t
        else:
            return None

    def set_slice(self, index, im):
        if self.check_slice_index(index):
            if self.__channel == 1:
                self.__data[index, :, :, 0] = im
            else:
                self.__data[index, :, :, :] = im

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

    def visualize(self, label=-1, min_depth=-1, max_depth=-1):
        if min_depth == -1:
            if label != -1:
                pts = self.__data.squeeze() == label
            else:
                pts = self.__data.squeeze()
        else:
            print(self.__data.shape)
            pts = np.empty((self.__data.shape[1], self.__data.shape[2], max_depth-min_depth+1), dtype=np.bool)
            for d in range(min_depth, max_depth+1):
                pts[:,:,d-min_depth] = self.get_slice(d, 0) == label
                '''if pts[:,:,d-min_depth].any():
                    fig = plt.figure()
                    plt.imshow(pts[:,:,d-min_depth], cmap='gray')
                    plt.show()'''
            #np.save('deneme.npy', pts)
        #pts = ndimage.binary_closing(pts, structure=np.ones((13, 13, 5)))
        verts, faces, normals, values = marching_cubes_lewiner(pts)
        mlab.triangular_mesh([vert[0] for vert in verts], [vert[1] for vert in verts], [vert[2] for vert in verts], faces)
        mlab.show()

    #This function creates an image frame for video recording
    def slice_2_frame(self, slice_index, feature_names=None, imgs=None, colors=None):
        img = imgs.get_slice(slice_index).astype(np.uint8)
        img = cv2.putText(img=img, text=f"{slice_index}", org=(25, 25), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,0), thickness=2)

        label_im = self.get_slice(slice_index, 0)
        labels = np.unique(label_im)
        for i in labels:
            if i == 0:# or i-1 not in self.features.index:
                continue
            temp = label_im == i
            if temp.any():
                color = colors[i-1, :]
                edge = find_edge(temp)
                bw_on_image(img, edge, color)
                xx = np.argwhere(temp)
                txtY, txtX = int(np.mean(xx[:, 0])), int(np.mean(xx[:, 1]))
                txt = "{}".format(i)
                '''if feature_names is not None:
                    txt += ","
                    for j in range(0, len(feature_names)):
                        feature_name = feature_names[j]
                        val = self.features.loc[i-1, feature_name]
                        if feature_name == 'surface_volume':
                            txt += "{:.2f}".format(val)
                        else:
                            txt += "{}".format(int(val))
                        if j != len(feature_names)-1:
                            txt += ","
                '''
                img = cv2.putText(img=img, text=txt, org=(txtX, txtY), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.5, color=color, thickness=2)
        return img

    def save_video(self, filename=None, feature_names=None, imgs=None):
        colors = np.random.randint(255, size=(10000, 3)).astype(np.float)
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), fps=5, frameSize=(self.__width, self.__height))
        for i in range(0, self.get_depth(), 1):
            frame = self.slice_2_frame(i, feature_names, imgs, colors)
            #print(frame.dtype)
            #frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
            #print(frame.dtype)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            '''if i%150 == 0:
                print(i)
                plt.imshow(frame)
                plt.show()'''
            out.write(frame)
        out.release()
