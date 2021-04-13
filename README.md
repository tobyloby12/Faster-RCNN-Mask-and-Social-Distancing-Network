# Faster-RCNN-Mask-and-Social-Distancing-Network
This is a network using the faster RCNN as a backbone and applying this to mask detection. It also has the ability to detect whether people are wearing face masks or not.

## network_again.py
This is the primary code I have created and has the functions used to change different aspects of the code. In here is the config files which can be changed to change values when training. It also has sections of commented out code at the bottom which have many functions used during the creating of visuals and evalutations.

## social_distancing.py
This is secondary code that was written and contains all the functions needed by network_again.py to create the social distancing aspect of the model. It uses openCV to perspective warp and plot out images for use as visual output.

## create_video_inference.py
This is a subset of code that allows for any video to be put in and for it to be processed to get 3 videos out which correspond to different areas of the network to see the different components working.

## train.py
This is where the training of the network can be found. The training of the dataset to be trained on needs to have a folder with 2 folders inside of test and train and within each of these folders there should be 2 folders for images and annotations each.

## image_aug.py
This is where the main image augmentation I used is found and can be rerun to create a seperate augmentation dataset.

## xml_test.py
This is where the functions used to create xml files are placed.