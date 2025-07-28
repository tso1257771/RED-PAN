import os
import cv2
import numpy as np
from glob import glob


image_folder = "./out_trg/fig_trg_20190704_WCS2_REDPAN_60s/"
video_name = "./20190704_WCS2_REDPAN_60s.avi"

images = np.sort(glob("./out_trg/fig_trg_20190704_WCS2_REDPAN_60s/*.png"))
frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width, height))

for image in images:
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()

os.system(f'ffmpeg -i {video_name} {video_name.replace("avi", "mp4")}')
