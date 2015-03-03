import cv2
import skimage.io
import numpy as np

def load_image_from_memory(data, color=True):
    img = cv2.imdecode(np.asarray(bytearray(data)), cv2.CV_LOAD_IMAGE_COLOR)    
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    img = skimage.img_as_float(img).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img
