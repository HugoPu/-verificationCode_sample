import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from config import Config as config
from utils.image_generate_utils import generate_artificial_image, get_code_image
from utils.image_process_utils import preprocess_src_image


test_var = None

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

def get_next_batch(batch_size,
                   config,
                   is_training):
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

    image_paths = tf.gfile.Glob(image_floder_path + '/*.jpg') + \
                  tf.gfile.Glob(image_floder_path + '/*.JPG')

    if is_training:
        batch_size = batch_size
    else:
        batch_size = len(image_paths)

    batch_x = np.zeros([batch_size, image_height * image_width])
    batch_y = np.zeros([batch_size, num_classes])

    for i in range(batch_size):
        rand_num = random.random()
        if rand_num > image_scale and is_training:
            text = gen_text(chars, min_num_chars, max_num_chars)
            image = generate_artificial_image(text, config)
        else:
            num_path = len(image_paths) - 1
            idx = random.randint(0, num_path)
            path = image_paths[idx]
            # path = '/sdb/hugo/data/pic/recognize/test/80613.jpg'
            image_paths.remove(path)
            text, image = get_code_image(path)
            image = preprocess_src_image(image, config)

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








