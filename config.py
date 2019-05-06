class Config(object):
    IMAGE_HEIGHT = 60
    IMAGE_WIDTH = 160
    MAX_NUM_CHARS = 6
    MIN_NUM_CHARS = 5
    CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    PATCH_CHAR = '_'
    TRAIN_FOLDER_PATH = '/sdb/hugo/data/pic/recognize/train/cut'
    TEST_FOLDER_PATH = '/sdb/hugo/data/pic/recognize/test'
    IMAGE_SCALE = 0.1
    CHAR_SET_LEN = len(CHARS) + 1
    NUM_CLASSES = CHAR_SET_LEN * MAX_NUM_CHARS
    OUTPUT = 'output'
    IMAGE_PREPROCESS = ['random_cut', 'threshold', 'random_rotate', 'random_padding']