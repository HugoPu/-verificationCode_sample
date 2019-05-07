import time
import numpy as np
import tensorflow as tf

import scipy.misc as misc

import training as tr

from utils.data_utils import vec2text , get_code_image, preprocess
from training import crack_captcha_cnn

from config import Config as config


def crack_image(image, config, sess, output):
    predict = tf.argmax(tf.reshape(output, [-1, config.MAX_NUM_CHARS, config.CHAR_SET_LEN]), 2)
    text_list = sess.run(predict, feed_dict={tr.X: [image], tr.keep_prob: 1})

    text = text_list[0].tolist()
    vector = np.zeros(config.NUM_CLASSES)
    i = 0
    for n in text:
        vector[i * config.CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector, config.CHAR_SET_LEN, config.PATCH_CHAR)


if __name__ == '__main__':
    start = time.clock()

    image_floder_path = config.TEST_FOLDER_PATH
    image_height = config.IMAGE_HEIGHT
    image_width = config.IMAGE_WIDTH

    image_paths = tf.gfile.Glob(image_floder_path + '/*.jpg') + \
                  tf.gfile.Glob(image_floder_path + '/*.JPG')

    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        correct_count = 0
        saver.restore(sess, tf.train.latest_checkpoint(config.OUTPUT))

        for path in image_paths:
            text, image = get_code_image(path)

            processed_image = preprocess(image, config, is_training=False)
            predict_text = crack_image(processed_image, config, sess, output)

            if text != predict_text:
                image.save(config.OUTPUT + '/' + text + '.jpg')
                processed_image_reshape = processed_image.reshape((60, 160))
                misc.imsave(config.OUTPUT + '/' + predict_text + '_processed.jpg', processed_image_reshape * 255)
            else:
                correct_count += 1

        print('accuracy:{}'.format(correct_count / len(image_paths)))