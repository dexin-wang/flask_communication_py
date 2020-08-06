import cv2
import torch
import PIL.PngImagePlugin
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import rotate, resize

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Image:
    """
    Wrapper around an image with some convenient functions.
    """
    def __init__(self, img):
        self.img = img

    def __getattr__(self, attr):
        # Pass along any other methods to the underlying ndarray
        return getattr(self.img, attr)

    @classmethod
    def from_file(cls, fname, bright):
        # return cls(imread(fname))
        img = imread(fname)
        # 随机亮度
        img = np.clip(img + bright, 0, 255)
        # img = img[:, :, [2, 1, 0]]
        return cls(img)

    def pre_process(self):
        """
        图像预处理（去躁）
        :return:
        """
        self.img = cv2.fastNlMeansDenoisingColored(self.img)

    def copy(self):
        """
        :return: Copy of self.
        """
        return self.__class__(self.img.copy())

    def crop(self, top_left, bottom_right, resize=None):
        """
        Crop the image to a bounding box given by top left and bottom right pixels.
        :param top_left: tuple, top left pixel.
        :param bottom_right: tuple, bottom right pixel
        :param resize: If specified, resize the cropped image to this size
        """
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        if resize is not None:
            self.resize(resize)

    def cropped(self, *args, **kwargs):
        """
        :return: Cropped copy of the image.
        """
        i = self.copy()
        i.crop(*args, **kwargs)
        return i

    def normalise(self):
        """
        Normalise the image by converting to float [0,1] and zero-centering
        """
        self.img = self.img.astype(np.float32)/255.0
        self.img -= self.img.mean()

    def resize(self, shape):
        """
        Resize image to shape.
        :param shape: New shape.
        """
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)

    def resized(self, *args, **kwargs):
        """
        :return: Resized copy of the image.
        """
        i = self.copy()
        i.resize(*args, **kwargs)
        return i

    def rotate(self, angle, center=None):
        """
        Rotate the image.
        :param angle: Angle (in radians) to rotate by.
        :param center: Center pixel to rotate if specified, otherwise image center is used.
        """
        if center is not None:
            center = (center[1], center[0])
        # 这里反转的原因是：输入的center是(row, col)，而skimage.transform.rotate的旋转中心的格式是(x, y),所以要反转
        self.img = rotate(self.img, angle/np.pi*180, center=center, mode='edge', preserve_range=True).astype(self.img.dtype)

    def rotated(self, *args, **kwargs):
        """
        :return: Rotated copy of image.
        """
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i

    def show(self, ax=None, **kwargs):
        """
        Plot the image
        :param ax: Existing matplotlib axis (optional)
        :param kwargs: kwargs to imshow
        """
        if ax:
            ax.imshow(self.img, **kwargs)
        else:
            plt.imshow(self.img, **kwargs)
            plt.show()

    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc: self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode='edge', preserve_range=True).astype(self.img.dtype)

    def zoomed(self, *args, **kwargs):
        """
        :return: Zoomed copy of the image.
        """
        i = self.copy()
        i.zoom(*args, **kwargs)
        return i


class DepthImage(Image):
    def __init__(self, img):
        super().__init__(img)

    @classmethod
    def from_pcd(cls, pcd_filename, shape, default_filler=0, index=None):
        """
            Create a depth image from an unstructured PCD file.
            If index isn't specified, use euclidean distance, otherwise choose x/y/z=0/1/2
        """
        img = np.zeros(shape)
        if default_filler != 0:
            img += default_filler

        with open(pcd_filename) as f:       # 默认只读
            for l in f.readlines():
                ls = l.split()

                if len(ls) != 5:
                    # Not a point line in the file.
                    continue
                try:
                    # Not a number, carry on.
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])          # index
                r = i // shape[1]       # row
                c = i % shape[1]        # col

                if index is None:
                    x = float(ls[0])    # x
                    y = float(ls[1])    # y
                    z = float(ls[2])    # z

                    img[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)   # 方案一
                    # img[r, c] = z       # 方案二 只用z

                else:
                    img[r, c] = float(ls[index])

        return cls(img/1000.0)      # 米变毫米

    @classmethod
    def pcd_to_depth(cls, pcd_filename, shape):
        """
        由pcd文件生成单通道深度图png
        :param pcd_filename:
        :param shape:图像的shape
        :return:
        """
        img = np.zeros(shape)   # (480, 640)

        with open(pcd_filename) as f:
            for l in f.readlines():
                ls = l.split()

                if len(ls) != 5:
                    # Not a point line in the file.
                    continue
                try:
                    # Not a number, carry on.
                    float(ls[0])
                except ValueError:
                    continue

                idx = int(ls[4])
                row = idx // shape[1]
                col = idx % shape[1]

                x = float(ls[0])  # x
                y = float(ls[1])  # y
                z = float(ls[2])  # z

                # img[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)     # 方案一
                img[row, col] = z       # 方案二 只用z

        return cls(img / 1000.0)  # 毫米变米

    @classmethod
    def from_tiff(cls, fname):
        return cls(imread(fname))

    def inpaint(self, missing_value=0):
        """
        填充缺失值
        :param missing_value: Value to fill in teh depth image.
        """
        border = 3
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        self.img = cv2.copyMakeBorder(self.img, border, border, border, border, cv2.BORDER_DEFAULT)     # 上下左右都镜像扩展一个像素，方便后面修复
        mask = (self.img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(self.img).max()
        self.img = self.img.astype(np.float32) / scale  # Has to be float32, 64 not supported.  归一化到[-1, 1]之间
        self.img = cv2.inpaint(self.img, mask, border, cv2.INPAINT_NS)       # 修复深度图

        self.img = self.img[border:int(-1*border), border:int(-1*border)]     # 去掉之前补充的边界
        self.img = self.img * scale

    def gradients(self):
        """
        Compute gradients of the depth image using Sobel filtesr.
        :return: Gradients in X direction, Gradients in Y diretion, Magnitude of XY gradients.
        """
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_DEFAULT)
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

        return DepthImage(grad_x), DepthImage(grad_y), DepthImage(grad)

    def normalise(self):
        """
        Normalise by subtracting the mean and clippint [-1, 1]
        """
        # self.img = np.clip((self.img - self.img.mean()), -1, 1)     # 减均值。小于-1的全部设为-1,1一样   原始

        mi = -1
        ma = 1
        # 缩放到[-1, 1]区间
        k = (ma - mi) / (np.max(self.img) - np.min(self.img))
        b = ma - k * np.max(self.img)      # b = y - k * x
        self.img = self.img * k + b

    def scale(self, val1, val2):
        """
        将self.img缩放到[val1, val2]之间
        :param val1:
        :param val2:
        :return:
        """
        # 缩放到[-1, 1]区间
        k = (val2 - val1) / (np.max(self.img) - np.min(self.img))
        b = val2 - k * np.max(self.img)  # b = y - k * x
        self.img = self.img * k + b


class WidthImage(Image):
    """
    A width image is one that describes the desired gripper width at each pixel.
    """
    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.  Also scales the width accordingly.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        super().zoom(factor)
        self.img = self.img/factor

    def normalise(self):
        """
        Normalise by mapping [0, 150] -> [0, 1]
        """
        self.img = np.clip(self.img, 0, 150.0)/150.0

def input_img(img_path):
    img_ = Image.from_file(img_path, 0)     # 读取文件
    img_.pre_process()      # 预处理 (去躁)
    img_.resize((224, 224)) # resize
    img_.normalise()        # 归一化
    img = img_.img.transpose((2, 0, 1))     # (H, W, C) -> (C, H, W)
    img = torch.from_numpy(np.expand_dims(img, 0).astype(np.float32))  # np转tensor
    return img

