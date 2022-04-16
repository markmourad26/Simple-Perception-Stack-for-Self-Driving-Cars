import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import math as m


def plot_imgs(imgs, layout='row', cols=2, figsize=(20, 12)):
    rows = m.ceil(len(imgs) / cols)
    f, ax = plt.subplots(figsize=figsize)
    if layout == 'row':
        for idx, data in enumerate(imgs):
            img, title = data
            channels_num = len(img.shape)
            plt.subplot(rows, cols, idx+1)
            plt.title(title, fontsize=20)
            plt.axis('off')
            if channels_num == 2:
                plt.imshow(img, cmap='gray')
            elif channels_num == 3:
                plt.imshow(img)
                
    elif layout == 'col':
        counter = 0
        for r in range(rows):
            for c in range(cols):
                img, title = imgs[r + rows*c]
                channels_num = len(img.shape)
                plt.subplot(rows, cols, counter+1)
                plt.title(title, fontsize=20)
                plt.axis('off')
                if channels_num == 2:
                    plt.imshow(img, cmap='gray')
                elif channels_num == 3:
                    plt.imshow(img)
              
                counter += 1
    return ax

def capture_frames(video_src, frames_dst):
    cap = cv2.VideoCapture(video_src)

    print('Converting video to sequence of frames...')
    
    count = 0
    success = True
    while success:
        success, frame = cap.read()
        cv2.imwrite(frames_dst + 'frame{:04}.jpg'.format(count), frame)
        count += 1

    print('Conversion completed!')
    
