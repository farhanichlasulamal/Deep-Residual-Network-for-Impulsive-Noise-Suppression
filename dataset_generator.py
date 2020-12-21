import os
import numpy as np

from PIL import Image
from multiprocessing import Pool

CROP_SIZE = (64, 64)
COLOR = True
SET = 'train'
DATASET_DIR = 'datasets'
SRC_PATH = os.path.join(DATASET_DIR, 'src')
FORMAT = 'PNG'

def crop(img_arr, block_size):
    h_b, w_b = block_size
    v_splited = np.vsplit(img_arr, img_arr.shape[0]//h_b)
    h_splited = np.concatenate([np.hsplit(col, img_arr.shape[1]//w_b) for col in v_splited], 0)
    return h_splited


def generate_patches(set_path, files, clean_path):
    img_path = os.path.join(set_path, files)
    img = Image.open(img_path)
    img = img.convert('RGB') if COLOR else img.convert('L')  

    name, format = files.split('.')
    clean_dir = os.path.join(clean_path, name)
    if not os.path.exists(clean_dir):
        os.mkdir(clean_dir)
    
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]

    if CROP_SIZE == None:
        clean = np.copy(img)
        clean_patches = np.expand_dims(clean, 0)
    else:
        rem_h = (h % CROP_SIZE[0])
        rem_w = (w % CROP_SIZE[1])
        clean = img[:h-rem_h, :w-rem_w]
        clean_patches = crop(clean, CROP_SIZE)

    for i in range(clean_patches.shape[0]):
        clean_img = Image.fromarray(clean_patches[i])
        clean_img.save(
            os.path.join(clean_dir, '{}.{}'.format(i, FORMAT))
        )


def main():
    print('[ Creating Dataset ]')
    print('Set       : {}'.format(SET))
    print('Format    : {}'.format(FORMAT))
    print('Color     : {}'.format(COLOR))
    print('Crop Size : {}'.format(CROP_SIZE))

    input('Press Enter to continue...')
    src_path = os.path.join(SRC_PATH, SET)
    set_path = os.path.join(DATASET_DIR, SET)
    if not os.path.exists(set_path):
        os.makedirs(set_path)

    clean_path = os.path.join(set_path, 'clean')
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    img_files = os.listdir(src_path)

    pool = Pool(32)
    for files in img_files:
        #generate_patches(src_path, files, clean_path)
        res = pool.apply_async(
            generate_patches,
            args=(src_path, files, clean_path, )
        )
        print(res)
    pool.close()
    pool.join()
    print('Dataset Created')


if __name__ == '__main__':
    main()
