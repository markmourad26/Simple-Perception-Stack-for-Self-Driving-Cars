import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy

def bin_spatial(img, size=(32, 32)):
    '''
    Performs spatial binning to reduce the image size and consequently the number
    image pixels
    :param img(ndarray): Image
    :param size((int, int)): New image size
    :return (ndarray): Updated image
    '''
    return cv2.resize(img, size).ravel()

def convert_color(img, cspace='HSV'):
    if cspace == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif cspace == 'LAB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        return np.copy(img)
    
def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Computes the histogram of color values
    :param img(ndarray): Image
    :param nbins(int): Number of histogram bins
    :param bins_range(int): Lower and upper range of the bins
    :return (ndarray): Histogram of color values
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     transform_sqrt=False, feature_vec=True):
    '''
    Wrapper function to compute the HOG (Histogram of Oriented Gradients) feature descriptor
    '''
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=transform_sqrt,
               feature_vector=feature_vec)


    
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    '''
    Function to extract features from a single image/video frame
    '''

    # Define an empty list to receive features
    features = []
    # Apply color conversion
    feature_img = convert_color(img, cspace=color_space) # returns in 0-255 range
    # Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_img, size=spatial_size)
        # Append features to list
        features.append(spatial_features)

    # Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_img, nbins=hist_bins)
        # 6) Append features to list
        features.append(hist_features)

    # Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_img.shape[2]):
                hog_features.extend(get_hog_features(feature_img[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     transform_sqrt=True,
                                                     feature_vec=True))
        else:
            hog_features = get_hog_features(feature_img[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block,
                                            transform_sqrt=True,
                                            feature_vec=True)

        # Append features to list
        features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(features)

def extract_features(imgs, params):
    '''
    Extracts user specified features (either Raw Color values/ Histogram of Color values/ HOG or a combination of these)
    from a list of images
    '''
    if not params:
        raise Exception('ERROR: Please provide a valid params dict!')
        
    features = []
    for img_path in imgs:
        img = mpimg.imread(img_path)
        
        features.append(single_img_features(
            img,
            color_space=params['color_space'],
            spatial_size=params['spatial_size'],
            hist_bins=params['hist_bins'],
            orient=params['orient'],
            pix_per_cell=params['pix_per_cell'],
            cell_per_block=params['cell_per_block'],
            hog_channel=params['hog_channel'],
            spatial_feat=params['spatial_feat'],
            hist_feat=params['hist_feat'],
            hog_feat=params['hog_feat']
        ))
        
    return features


# searching for detections, Handling multiple detections and false positives
def find_cars(img, ystart, ystop, scale, svc, scaler, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins, cspace):
    '''
    HOG Sub-sampling window search
    
    This function builds upon on the Sliding Window approach but instead of computing the expensive
    HOG features for every multi-scale window, it extracts the HOG features just once and subsamples it
    to search for detections over all the different multi-scale windows within the search area.
    Each window is defined by a scaling factor where a scale of 1 would result in a window that's
    8x8 cells and each cell has 8x8 pixels. The overlap of each window is in terms of the cell distance.
    This means that a cells_per_step = 2 would result in a search window overlap of 75%.
    
    :param img(ndarray): Frame
    :param ystart(int): Search area low y-coordinate 
    :param ystop(int): Search area high y-coordinate 
    :param scale(float): Window scale
    :param svc(LinearSVC): Instance of the trained Linear SVC 
    :param scaler(StandardScaler): Instance of the Standard Scaler fitted over the training set
    :param orient(int): Number of orientation bins; param for extracting HOG features
    :param pix_per_cell(int): Number of pixels per cell; param for extracting HOG features
    :param cell_per_block(int): Number of cells per block; param for extracting HOG features
    :param spatial_size((int, int)): Spatial bin size; param for extracting raw color values
    :param hist_bins(int): Number histogram bins; param for extracting Histogram of color values
    :param cspace(string): Color space in which to extract the features
    :return : Tuple (List of bounding boxes of detections, Image with the detection bounding boxes drawn,
                        Image with all the search windows drawn)
    '''
    
    # Note: The number of features are maintained across the multi-scaled windows by:
    # 1. Making sure the window size = size of the image that we trained with i.e. 64x64
    # 2. By resizing the image to achieve the effect of larger/smaller window
    
    detections_img = np.copy(img)
    draw_wins_img = np.copy(img)
    bbox_list = []
    
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, cspace=cspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps. Note: cells > blocks > windows
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
#     nfeat_per_block = orient*cell_per_block**2
    
    window = 64 # 64 is the orginal sampling rate, with 8 cells and 8 pix per cell
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            box_left = np.int(xleft*scale)
            box_top = np.int(ytop*scale)
            win = np.int(window*scale)
            bbox = ((box_left, box_top + ystart), (box_left + win, box_top + win + ystart))

            cv2.rectangle(draw_wins_img, bbox[0], bbox[1], (0, 0, 255), 6) 
            
            # Try the decision_function here...
            if test_prediction == 1:               
                bbox_list.append(bbox)
                cv2.rectangle(detections_img, bbox[0], bbox[1], (0, 0, 255), 6) 
                
    return bbox_list, detections_img, draw_wins_img

def add_heat(heatmap, bbox_list):
    '''
    Function to add heat to heatmap for a list of bounding boxes
    '''
    
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap
    
def apply_threshold(heatmap, threshold):
    '''
    Function that zeros out all the pixels in a heatmap that are below a user defined threshold
    '''
    
    heatmap[heatmap <= threshold] = 0
    return heatmap

def add_heat_and_threshold(img, bboxes):
    heatmap = np.zeros_like(img[:,:,0], dtype=np.float)
    heatmap = add_heat(heatmap, bboxes)
    heatmap = apply_threshold(heatmap, 3)
    heatmap = np.clip(heatmap, 0, 255)
    labels = label(heatmap)
    _, thresh_bboxes = draw_labeled_bboxes(np.copy(img), labels)
    return thresh_bboxes, heatmap

def draw_labeled_bboxes(img, labels):
    '''
    Function that takes in the labelled detections in a heatmap and puts bounding
    boxes in the original image around the labelled regions.
    '''
    
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        bboxes.append(bbox)
        
    return img, bboxes

