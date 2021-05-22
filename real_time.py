from ximea import xiapi
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(34000)
cam.set_downsampling('XI_DWN_4x4')
# create instance of Image to store image data and metadata
img = xiapi.Image()

gaussian_kernel =  [[0,  0,  0,  0,  0,  0, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0, -1, -2, -3, -2, -1, 0],
                    [0,  0,  0,  0,  0,  0, 0],]

ndimage.gaussian_filter(gaussian_kernel, sigma=1)
# Define Lasca filter

def gaussian(imarray, wsize = 11):
    immean = ndimage.uniform_filter(ndimage.convolve(imarray,gaussian_kernel), size=wsize)
    im2mean = ndimage.uniform_filter(np.square(ndimage.convolve(imarray,gaussian_kernel)), size=wsize)
    imcontrast = np.sqrt(im2mean / np.square(immean) - 1)
    return imcontrast

# start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

try:
    print('Starting video. Press CTRL+C to exit.')
    t0 = time.time()
    while True:
        # get data and pass them from camera to img
        cam.get_image(img)

        # create numpy array with data from camera. Dimensions of the array are
        # determined by imgdataformat
        data = img.get_image_data_numpy()
        # Lasca
        imarray = np.array(data).astype(float)
        # make contrast with window of 5 pixels
        imcontrast05 = gaussian(imarray, 11)
        # show acquired image with time since the beginning of acquisition
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = '{:5.2f}'.format(time.time() - t0)
        cv2.putText(
            data, text, (900, 150), font, 4, (255, 255, 255), 2
        )



        cv2.imshow('XiCAM example', imcontrast05)

        cv2.waitKey(1)

except KeyboardInterrupt:
    cv2.destroyAllWindows()

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print('Done.')
