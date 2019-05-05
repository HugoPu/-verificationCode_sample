import os
import pygame
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.filters import threshold_local

from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw, ImageFont

from config import Config as config

test_var = None

def threshold(image_np):
    thresh = threshold_local(image_np, block_size=35)
    binary = image_np > thresh
    return binary

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text, max_num_chars, char_set_len, patch_char):
    text_len = len(text)
    if text_len > max_num_chars:
        raise ValueError('')
    elif text_len < max_num_chars:
        text = text + patch_char

    vector = np.zeros(max_num_chars * char_set_len)

    def char2pos(c):
        if c == patch_char:
            k = char_set_len - 1
            return k
        k = ord(c) - 48
        return k

    for i, c in enumerate(text):
        idx = i * char_set_len + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec, char_set_len, patch_char):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % char_set_len
        if char_idx < 10:
            char_code = char_idx + ord('0')
            text.append(chr(char_code))
        # else:
        #     char_code = ord(patch_char)

    return "".join(text)

def gen_text(chars, min_num_chars, max_num_chars):
    text_list = []

    captcha_size = random.randint(min_num_chars, max_num_chars)
    for i in range(captcha_size):
        c = random.choice(chars)
        text_list.append(c)

    text = ''.join(text_list)
    return text

def gen_normal_text_image(text):
    image_name = 'temp.jpg'
    pygame.init()
    font = pygame.font.Font('DFFK_S3.TTC', 64)
    ftext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(ftext, image_name)
    image = Image.open(image_name)
    return image

def gen_captcha_image(text):
    image = ImageCaptcha(fonts=['DFFK_S3.TTC'])
    # image = ImageCaptcha()
    captcha = image.generate(text)
    captcha_image = Image.open(captcha)
    return captcha_image

def random_cut_image(image, text):
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


def get_code_image(path):
    image = Image.open(path)
    text = os.path.basename(path).split('.')[0]

    return text, image

def preprocess(image, config):
    image = image.resize(
        (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), Image.BILINEAR)
    image = np.array(image)
    image = convert2gray(image)
    if 'threshold' in config.IMAGE_PREPROCESS:
        image = threshold(image)

    image = image .flatten()

    return image

def test_padding(image):
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

# 生成一个训练batch
def get_next_batch(batch_size,
                   config,
                   is_training=True):
    image_height = config.IMAGE_HEIGHT
    image_width = config.IMAGE_WIDTH
    image_floder_path = config.TRAIN_FOLDER_PATH if is_training else config.TEST_FOLDER_PATH
    chars = config.CHARS
    min_num_chars = config.MIN_NUM_CHARS
    max_num_chars = config.MAX_NUM_CHARS
    char_set_len = config.CHAR_SET_LEN
    patch_char = config.PATCH_CHAR
    image_scale = config.IMAGE_SCALE
    num_classes = config.NUM_CLASSES

    batch_x = np.zeros([batch_size, image_height * image_width])
    batch_y = np.zeros([batch_size, num_classes])

    image_paths = tf.gfile.Glob(image_floder_path + '/*.jpg') + \
                  tf.gfile.Glob(image_floder_path + '/*.JPG')

    for i in range(batch_size):
        rand_num = random.random()
        if rand_num> image_scale and is_training:
            text = gen_text(chars, min_num_chars, max_num_chars)

            if random.random() - image_scale > (1 - image_scale) / 2:
            # if True:
                image = gen_normal_text_image(text)
            else:
                image = gen_captcha_image(text)

            image = test_padding(image)

            if 'random_cut' in config.IMAGE_PREPROCESS:
                image = random_cut_image(image, text)

        else:
            path = random.sample(image_paths, 1)[0]
            text, image = Image.open(path)

        image = preprocess(image, config)

        batch_x[i, :] = image  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text, max_num_chars, char_set_len, patch_char)
        if __name__ == '__main__':
            global test_var
            test_var = text

    return batch_x, batch_y

if __name__ == '__main__':
    for i in range(10):
        batch_x, batch_y = get_next_batch(1, config)

        image = np.reshape(batch_x[0], (config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

        converted_text = vec2text(batch_y[0], len(config.CHARS) + 1, config.PATCH_CHAR)
        assert converted_text == test_var

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, converted_text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)

        plt.show()

        a = 1

    # from PIL import Image, ImageDraw, ImageFont, ImageFilter
    #
    # # 240 x 60:
    # width = 160
    # height = 60
    # image = Image.new('RGB', (width, height), (255, 255, 255))
    # # 创建Font对象:
    # font = ImageFont.truetype(size =  36)
    # # font = None
    # # 创建Draw对象:
    # draw = ImageDraw.Draw(image)
    # text = '123456'
    # position = (0, 0)
    # draw.text(position, text, font=font, fill="#000000", spacing=0, align='left')
    # image.show()





    # import os
    # # import Image, ImageDraw, ImageFont, ImageFilter
    # import random
    #
    # BASE_DIR = os.path.dirname(os.getcwd())
    #
    # text = "123456"
    #
    # # PIL实现
    # width = 60 * 4
    # height = 60 * 2
    # im = Image.new('RGB', (width, height), (255, 255, 255))
    # dr = ImageDraw.Draw(im)
    # font = ImageFont.truetype(os.path.join('fonts', BASE_DIR + "\\resources\\minijson.ttf"), 20)
    # font=None
    # dr.text((10, 5), text, font=font, fill='#000000')
    # im.show()









