"""
Car Detection pipeline

Usage:
    main.py [--debug] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--debug                                 debug mode on
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import time
import glob
from collections import deque
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from docopt import docopt
from utils import *

class Tracker(object):
    def __init__(self):
        '''
        Wrapper class that processes each video frame by:
        1. Extracting features (Raw color values/ Histogram of color values/ HOG)
        2. Searching for detection
        3. Handling multiple detections and false positives
        4. Tracking the detections frame by frame and reducing the jitter in detections by looking
        the past history of detections.
        '''
        
        self.nb_frames = 0 # Number of processed frames
        self.cache = deque(maxlen=12) # Cache to store the detections over the past 12 frames
        self.new_ystop = 0 # New vertical stop limit for the search area
        self.curr_bboxes = [] # List of bounding boxes for the current detections
        
    def process_frame(self, img, debug=False): 
        '''
        Function that:
        * Performs HOG Sub-sampling window search every 12 frames i.e. ~ 0.5 secs, and a,
        * Reduced window search every 6 frames which only scans the region of interest
          at where vehicle were previously detected.
          
        :param img(ndarray): Frame
        :return (ndarray): Processed frame with the higlighted detections
        '''
        
        heatmap = np.zeros_like(img[:,:,0], dtype=np.float)
        wins_img = np.copy(img)
        
        # Define section of the image where the search needs to be performed
        ystarts = [400, 400, 400, 400]
        ystops = [496, 528, 592, 656]
        # Define multi-scale windows for searching vehicles
        scales = [1., 1.25, 1.5, 1.75]
        
        bboxes_list = []
        if self.nb_frames % 12 == 0:
            for scale, ystart, ystop in zip(scales, ystarts, ystops):
                bboxes, detections_img, wins_img = find_cars(img, ystart, ystop, scale,
                                                        svc, scaler, orient, pix_per_cell,
                                                        cell_per_block, spatial_size, hist_bins, cspace)
                bboxes_list.extend(bboxes)

            thres_bboxes, heatmap = add_heat_and_threshold(img, bboxes_list)
            out = self.draw_detections(img, thres_bboxes)
        
        elif (self.nb_frames % 6 == 0) and (self.new_ystop > 0):
            ystops = [self.new_ystop if y > self.new_ystop else y for y in ystops]
            for scale, ystart, ystop in zip(scales, ystarts, ystops):
                bboxes, detections_img, wins_img = find_cars(img, ystart, ystop, scale,
                                                        svc, scaler, orient, pix_per_cell,
                                                        cell_per_block, spatial_size, hist_bins, cspace)
                bboxes_list.extend(bboxes)

            thres_bboxes, heatmap = add_heat_and_threshold(img, bboxes_list)
            out = self.draw_detections(img, thres_bboxes)
            
        else:
            thres_bboxes, heatmap = add_heat_and_threshold(img, self.curr_bboxes)
            out = draw_boxes(img, self.curr_bboxes)
        
        self.nb_frames += 1
        if debug:
            heatmap = np.dstack([heatmap, heatmap, heatmap]) * 255
            ret2 = np.hstack([img, wins_img])
            ret3 = np.hstack([heatmap, out])
            out = np.vstack([ret2, ret3])
            return out
        else:    
            return out
    
    def draw_detections(self, img, thres_bboxes):
        '''
        Draws the bounding boxes for the detections and reduces the jitter
        by leveraging the history of detections over the past 12 frames
        :param (ndarray): Frame
        :param thresh_boxes(List of Tuples): List of bounding boxes for the detections
                                            in the current frame
        :return (ndarray): Frame with the detections higlighted
        '''
        self.cache.append(thres_bboxes)
        heatmap = np.zeros_like(img[:,:,0], dtype=np.float)
        
        for bboxes in self.cache:
            heatmap = add_heat(heatmap, bboxes)
            
        heatmap = apply_threshold(heatmap, (len(self.cache) // 3)+1)
        heatmap = np.clip(heatmap, 0, 255)
        labels = label(heatmap)
        draw_img, bboxes = draw_labeled_bboxes(np.copy(img), labels)
        
        if np.array(bboxes).any():
            self.curr_bboxes = bboxes
            self.new_ystop = np.amax(np.array(bboxes), axis=0)[1,1] + 64
        else:
            self.new_ystop = 0

        return draw_img
    
def main():

    tracker = Tracker()
    
    args = docopt(__doc__)
    input = args['INPUT_PATH']
    video_output = args['OUTPUT_PATH']
    process_frame = lambda frame: tracker.process_frame(frame, debug=args['--debug'])
    video_input = VideoFileClip(input).subclip(20, 50)
    t1 = time.time()
    video_clip = video_input.fl_image(process_frame)
    video_clip.write_videofile(video_output, audio=False)
    print(f'Time = {(time.time() - t1)/60}')


if __name__ == "__main__":
    # Load pickled data
    data = pickle.load(open('classifier_data.p', mode='rb'))
    svc = data['clf']
    scaler = data['scaler']
    orient = data['orient']
    pix_per_cell = data['pix_per_cell']
    cell_per_block = data['cell_per_block']
    spatial_size = data['spatial_size']
    hist_bins = data['hist_bins']
    cspace = data['color_space']
    main()