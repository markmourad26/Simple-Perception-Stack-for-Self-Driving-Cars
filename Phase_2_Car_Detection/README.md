# Simple-Perception-Stack-for-Self-Driving-Cars

Phase 2
---

The goals / steps of this project are the following:

The various steps invovled in the pipeline are as follows:

* Perform feature extraction on a labeled training set of images and train a **Linear SVM classifier**. The feature vector consists of: 
  * **Histogram of Oriented Gradients (HOG)** 
  * Spatially binned raw color values, and,
  * Histogram of color values
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images
* Create a heat map of recurring detections frame by frame to reject outliers, handle multiple detections and follow detected vehicles
* Estimate a bounding box for vehicles detected


The image for the dataset is stored in the folder called `data`.  The images in `test_images` are for testing the pipeline on single frames.

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
```
#### 2.2 Using shell
Window: `bash main.sh input_path output_path --debug0/1`

Linux, MacOS: `./main.sh input_path output_path --debug0/1`
