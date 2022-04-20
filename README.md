# Simple-Perception-Stack-for-Self-Driving-Cars

The Project
---

The goals / steps of this project are the following:

The various steps invovled in the pipeline are as follows, each of these has also been discussed in more detail in the sub sections below:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.

## Usage:

### 1. Set up the environment 
#### 1.1 Using conda
`conda env create -f environment.yml`

To activate the environment:

Window: `conda activate car_env`

Linux, MacOS: `source activate car_env`

#### 1.2 Using pip
Window: `pip install -r requirements.txt`

Linux, MacOS: `pip install -r requirements.txt`

### 2. Run the pipeline:
#### 2.1 Using command window
```bash
python main.py INPUT_IMAGE OUTPUT_IMAGE_PATH
python main.py --debug INPUT_IMAGE OUTPUT_IMAGE_PATH
python main.py --video INPUT_VIDEO OUTPUT_VIDEO_PATH
python main.py --video --debug INPUT_VIDEO OUTPUT_VIDEO_PATH
```
#### 2.2 Using shell
Window: `bash main.sh input_path output_path --video0/1 --debug0/1`

Linux, MacOS: `./main.sh input_path output_path --video0/1 --debug0/1`
