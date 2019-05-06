import os
import pygame

from PIL import Image
from captcha.image import ImageCaptcha

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

def get_code_image(path):
    image = Image.open(path)
    text = os.path.basename(path).split('.')[0]

    return text, image