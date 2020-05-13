import cv2
import numpy as np
import random
from skimage.color import label2rgb
from sklearn.cluster import KMeans
from scipy.spatial.qhull import ConvexHull
from skimage import morphology
from skimage.filters import threshold_otsu, threshold_triangle, threshold_yen
from skimage.measure import marching_cubes_lewiner, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_fill_holes
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt

stain_mtx = np.array([[0.6443186,  0.09283127, 0.63595447], [0.71667568, 0.95454569, 0.], [0.26688857, 0.28323998, 0.77172658]])


def bw_on_image(im, bw, color):
    for i in range(0, 3):
        t = im[:, :, i]
        t[bw] = color[i]
        im[:, :, i] = t


def label_on_image(im, labelIm, M=None, inPlace=True, randSeed=None):
    if not inPlace:
        im2 = im.copy()
        label_on_image(im2, labelIm, M, True)
        return im2
    else:
        max_label = np.max(labelIm)
        if M is None:
            if randSeed is not None:
                random.seed(randSeed)
            M = np.random.randint(0, 255, (max_label+1, 3))

        if max_label == 0:
            return
        elif max_label == 1:
            bw_on_image(im, labelIm == max_label, M[1, :])
        else:
            for r in range(1, im.shape[0]-1):
                for c in range(1, im.shape[1]-1):
                    l = labelIm[r, c]
                    #if l > 0 and (l != labelIm[r,c-1] or l != labelIm[r,c+1] or l != labelIm[r-1,c] or l != labelIm[r+1,c]):
                    if l > 0:
                        im[r,c,0] = M[l, 0]
                        im[r,c,1] = M[l, 1]
                        im[r,c,2] = M[l, 2]


def find_edge(bw, strel=5):
    return np.bitwise_xor(bw, binary_erosion(bw, structure=np.ones((strel, strel))).astype(bw.dtype))


def get_stain_mtx():
    stain_mtx = np.array([[0.644211, 0.716556, 0.266844],[0.092789, 0.954111, 0.283111],[0, 0, 0]]).transpose()
    temp = np.sqrt(np.sum(np.square(stain_mtx),0))
    temp[np.where(temp==0)] = 1e-200
    for i in range(0, stain_mtx.shape[0]):
        stain_mtx[i,:] = np.multiply(stain_mtx[i,:], 1. / temp)

    if np.all(stain_mtx[:,1]==0):
        stain_mtx[:,1] = stain_mtx[[2, 0, 1],0]

    if np.all(stain_mtx[:,2]==0):
        temp0 = np.square(stain_mtx)
        temp = np.sum(temp0,1)
        temp[np.where(temp>1)] = 1
        stain_mtx[:,2] = np.sqrt(1.0 - temp)
        stain_mtx[:,2] = np.divide(stain_mtx[:,2],np.sqrt(np.sum(np.square(stain_mtx[:,2]))))
    return stain_mtx


def color_deconvolution(img):
    img[np.where(img == 0)] = 10e-20
    A = -np.log(img/255.)
    A = A.reshape((A.shape[0]*A.shape[1],3))
    A = np.linalg.lstsq(stain_mtx, A.transpose(), rcond=None)[0]
    A = A.transpose().reshape(img.shape)
    A = np.round(255*np.exp(-A))
    A[np.where(A>255)] = 255
    return A


def calculate_threshold_automatically(im, threshold_func):
    try:
        if threshold_func == 'Otsu':
            thresh = np.max([5.0, threshold_otsu(im)])
        elif threshold_func == 'Triangle':
            thresh = threshold_triangle(im)
        elif threshold_func == 'Yen':
            thresh = threshold_yen(im)
        else:
            print("Unsupported threshold method! Returning 5.0...")
            thresh = 5.
    except:
        thresh = 5.0
    return thresh


def threshold(im, const=-1, reverseFlag=False, threshold_func='Otsu'):
    if im is not None:
        if const == -1:
            thresh = calculate_threshold_automatically(im, threshold_func)
        else:
            if isinstance(const, np.integer):
                thresh = const
            else:
                thresh = calculate_threshold_automatically(im, threshold_func) * const
        if reverseFlag:
            im = im <= thresh
        else:
            im = im > thresh
    return im


def otsu(im):
    thresh = 0
    if im is not None:
        try:
            thresh = threshold_otsu(im.reshape(-1))
        except ValueError:
            thresh = 0
    return thresh


def eliminate_small_area(bw, th, in_place=False):
    if in_place:
        remove_small_objects(bw, th, in_place=in_place)
    else:
        return remove_small_objects(bw, th, in_place=in_place)


def eliminate_small_holes(bw, in_place=False):
    if in_place:
        remove_small_holes(bw, in_place=in_place)
        return None
    else:
        bw = remove_small_holes(bw, in_place=in_place)
        return bw

def bw_close(im, strel, iter=1):
    pad_length = strel*2
    im = np.pad(im, pad_length, 'symmetric')
    #im = ndimage.binary_closing(im, structure=np.ones((strel,strel)), iterations=iter)
    morphology.binary_closing(im, selem=np.ones((strel, strel), dtype=im.dtype), out=im)
    return im[pad_length:-pad_length, pad_length:-pad_length]


def bw_fill(im):
    return binary_fill_holes(im)


def bw_convex_hull(im):
    #return morphology.convex_hull.convex_hull_image(im)
    return morphology.convex_hull.convex_hull_object(im)


def distance_transform(im):
    return distance_transform_edt(im)


def bw_erode(im, strel):
    pad_length = strel*2
    im = np.pad(im, pad_length, 'symmetric')
    morphology.binary_erosion(im, selem=np.ones((strel, strel), dtype=im.dtype), out=im)
    return im[pad_length:-pad_length, pad_length:-pad_length]


def create_mesh(data, spacing=(1,1,1)):
    return marching_cubes_lewiner(data, spacing=spacing)


def convexhull_volume(data, offset=10):
    points = np.ndarray(shape=(0,3), dtype=int)
    for i in range(0, data.shape[0], offset):
        for j in range(0, data.shape[1], offset):
            for k in range(0, data.shape[2], offset):
                if data[i][j][k] > 0:
                    points = np.append(points, [[i,j,k]], axis=0)
    print(points.shape)
    hull = ConvexHull(points)
    return hull


'''nuclei_vol.normalize()
    tissue_vol.normalize()
    indices = [0, 800]
    kmeansOTLS_slices(tissue_vol.get_slices(indices), nuclei_vol.get_slices(indices), k=5)
    return 0'''
def kmeansOTLS_slices(nuclei, cytoplasm, k=3):
    he, wi, de = nuclei.shape
    nucleir = nuclei.ravel()
    cytoplasmr = cytoplasm.ravel()
    data = np.vstack((nucleir, cytoplasmr)).T
    print(data.shape)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    # Plot whitened data and cluster centers in red
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], c='r')
    plt.show()

    labels = kmeans.labels_
    labels = np.reshape(labels, (he, wi, de))
    for i in range(0, de):
        nuc = nuclei[:,:,i]
        cyt = cytoplasm[:,:,i]
        res = np.dstack((nuc, cyt, nuc)).astype(np.uint8)
        label_on_image(res, labels[:,:,i], randSeed=30)
        plt.imshow(res)
        plt.show()


def print_label_on_image(res, labelIm):
    res = label_on_image(res, labelIm, inPlace=False)
    props = regionprops(labelIm)
    for region in props:
        res = cv2.putText(img=res, text=f"{region.label}", org=(int(region.centroid[0]), int(region.centroid[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,0,0), thickness=1)
    plt.imshow(res)
    plt.show()
