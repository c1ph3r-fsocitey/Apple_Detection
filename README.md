# Apple_Detection

## Steps to follow to make sure it works!

### Step 1 (Clone the repository)
clone the repository using git clone </br>`https://github.com/c1ph3r-fsocitey/Apple_Detection`

### Step 2 (Downloading Test Data and yoloV3 Model)
Download the testing data and yoloV3 file from [this link](https://drive.google.com/drive/folders/1EQ_-72dGQIOrpN2LXOFmhGFQCWEUwrll?usp=sharing)

### Step 3

make sure to put the downloaded files in the same folder (Apple_Detection)

## Testing

### Test 1 (for Picture)

use the following command to run the program for image apple detection

`python apple_detection_from_photo.py -i apple_tree_images_and_vids/apple-1.jpg -c yolov3.cfg -cl yolov3.txt -w yolov3.weights`

![Screenshot from 2023-08-18 00-07-09](https://github.com/c1ph3r-fsocitey/Apple_Detection/assets/109020327/6f0bfa7c-ce82-4974-b93c-5dee3ce56711)

### Test 2 (for Videos)

use the following command to run the program for video apple detection

`python apple_detection_from_video.py -v apple_tree_images_and_vids/apple_vid.mp4 -c yolov3.cfg -cl yolov3.txt -w yolov3.weights`

![apple_detecion](https://github.com/c1ph3r-fsocitey/Apple_Detection/assets/109020327/91970963-4a22-41ed-8c92-85fe4d19d2cb)

### ENJOY!!
