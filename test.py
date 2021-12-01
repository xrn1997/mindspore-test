import os
import cv2
from logzero import logger
from skimage import io

if __name__ == '__main__':
    file_path = "datasets/RP2K_Data/train"
    for root, dirs, files in os.walk(file_path):
        for f in files:
            logger.info(f)
            image = io.imread(os.path.join(root, f))
            logger.info(image.size)
            # image = io.imread(f)
            # image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            # cv2.imencode('.png', image)[1].tofile(f)
