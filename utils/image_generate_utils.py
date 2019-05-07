import os
import pygame
import random

from PIL import Image
from captcha.image import ImageCaptcha

from utils.image_process_utils import preprocess_generated_image

def _gen_normal_text_image(text):
    image_name = 'temp.jpg'
    pygame.init()
    font = pygame.font.Font('utils/DFFK_S3.TTC', 64)
    ftext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(ftext, image_name)
    image = Image.open(image_name)
    return image

def _gen_captcha_image(text):
    image = ImageCaptcha(fonts=['utils/DFFK_S3.TTC'])
    captcha = image.generate(text)
    captcha_image = Image.open(captcha)
    return captcha_image

def get_code_image(path):
    image = Image.open(path)
    text = os.path.basename(path).split('.')[0]

    return text, image

def generate_artificial_image(text, config):
    rand_num = random.random()
    if rand_num > 0.5:
        image = _gen_normal_text_image(text)
    else:
        image = _gen_captcha_image(text)

    image = preprocess_generated_image(image, text, config)

    return image
