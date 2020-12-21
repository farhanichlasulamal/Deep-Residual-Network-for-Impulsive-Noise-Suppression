import numpy as np
from sporco.util import spnoise

img = np.zeros([64, 64], dtype='uint8') + 128
img = spnoise(img, 0.1, 0, 255)
print(type(img[0][0]))