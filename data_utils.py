import os
import numpy as np
import random
import tensorflow as tf

from skimage.filters import threshold_local

from captcha.image import ImageCaptcha
from PIL import Image

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


# 生成字符对应的验证码
def gen_captcha_text_and_image(chars, min_num_chars, max_num_chars):
    image = ImageCaptcha()

    captcha_text = []

    captcha_size = random.randint(min_num_chars, max_num_chars)
    for i in range(captcha_size):
        c = random.choice(chars)
        captcha_text.append(c)

    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

def get_code_image(path, image_width, image_height):
    image = Image.open(path)
    image = image.resize((image_width, image_height), Image.BILINEAR)

    text = os.path.basename(path).split('.')[0]
    image = np.array(image)
    return text, image

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

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(chars, min_num_chars, max_num_chars)
            if image.shape == (60, 160, 3):
                return text, image

    image_paths = tf.gfile.Glob(image_floder_path + '/*.jpg') + \
                  tf.gfile.Glob(image_floder_path + '/*.JPG')

    for i in range(batch_size):
        if random.random() > image_scale and is_training:
            text, image = wrap_gen_captcha_text_and_image()
        else:
            text, image = get_code_image(
                random.sample(image_paths, 1)[0],
                image_width,
                image_height)

        image = convert2gray(image)
        image = threshold(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
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

        # f = plt.figure()
        # ax = f.add_subplot(111)
        # ax.text(0.1, 0.9, converted_text, ha='center', va='center', transform=ax.transAxes)
        # plt.imshow(image)
        #
        # plt.show()






