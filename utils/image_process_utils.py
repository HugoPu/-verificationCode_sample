import random
import cv2
import numpy as np

from PIL import Image
from skimage.filters import threshold_local
from skimage import color

from utils.image_generate_utils import *

def add_padding(image):
    """
    Add white padding and black padding to the image
    :param image:
    :return:
    """
    image_np = np.asarray(image)
    channel_one = image_np[:,:,0]
    channel_two = image_np[:, :, 0]
    channel_three = image_np[:, :, 0]

    channel_one = np.pad(channel_one, (((0, 0), (10, 10))), 'constant', constant_values=(255, 255))
    channel_two = np.pad(channel_two, (((0, 0), (10, 10))), 'constant', constant_values=(255, 255))
    channel_three = np.pad(channel_three, (((0, 0), (10, 10))), 'constant', constant_values=(255, 255))

    channel_one = np.pad(channel_one, (((10, 10), (5, 5))), 'constant', constant_values=(0, 0))
    channel_two = np.pad(channel_two, (((10, 10), (5, 5))), 'constant', constant_values=(0, 0))
    channel_three = np.pad(channel_three, (((10, 10), (5, 5))), 'constant', constant_values=(0, 0))

    image_np = np.dstack((channel_one, channel_two, channel_three))
    image = Image.fromarray(image_np)

    return image

def random_cut_image(image, text):
    """
    Random cut image edge
    :param image:
    :param text:
    :return:
    """
    if len(text) == 5:
        cut_width = image.width / 16
        cut_height = image.height / 5
    else:
        cut_width = image.width / 20
        cut_height = image.height / 5

    x1 = cut_width * random.random()
    x2 = image.width - cut_width * random.random()
    y1 = cut_height * random.random()
    y2 = image.height - cut_height * random.random()

    return image.crop((x1,y1,x2,y2))

def random_rotate(image):
    angle = (random.random() - 0.5) * 2 * 20
    im2 = image.convert('RGBA')
    rot = im2.rotate(angle, expand=1)
    fff= Image.new('RGBA', rot.size, (255,)*4)
    out = Image.composite(rot, fff, rot)
    return out

    # return Image.fromarray(new_image)

def threshold(image_np, config, is_trainging):
    image_np = (image_np * 255).astype(np.int)
    if is_trainging:
        offset = random.random() * 20
        thresh = threshold_local(
            image_np, block_size=config.BLOCK_SIZE, offset=offset)
    else:
        thresh = threshold_local(
            image_np, block_size=config.BLOCK_SIZE, offset=config.OFFSET)

    binary = image_np > thresh
    return binary


def convert2gray(img):
    # image_np = np.asarray(img,dtype='int')
    # for width in image_np:
    #     for height in width:
    #         if np.var(height[:]) > 500:
    #             height[:] = [255] * 3
    #
    # img = Image.fromarray(image_np)

    if len(img.shape) > 2:
        return color.rgb2gray(img)
    else:
        return img

def preprocess(image, text, config, is_training=True):
    if is_training:
        if 'add_padding' in config.IMAGE_PREPROCESS:
            add_padding(image)
        if 'random_rotate' in config.IMAGE_PREPROCESS:
            random_rotate(image)
        if 'random_cut' in config.IMAGE_PREPROCESS:
            random_cut_image(image, text)
        image = image.resize(
            (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), Image.BILINEAR)
        image = np.array(image)
        image = convert2gray(image)
        if 'threshold' in config.IMAGE_PREPROCESS:
            image = threshold(image, config)
    else:
        image = image.resize(
            (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), Image.BILINEAR)
        image = np.array(image)
        image = convert2gray(image)
        if 'threshold' in config.IMAGE_PREPROCESS:
            image = threshold(image, config, is_training)


    image = image .flatten()

    return image

if __name__ == '__main__':
    image = gen_normal_text_image('123456')
    new_image = random_rotate(image)
    new_image.show()
    a = 1