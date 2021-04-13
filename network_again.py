from os import listdir
import os
import tensorflow as tf
import pandas as pd
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import time
from social_distancing import *


class MaskDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True, prefix = ''):
        # define one class
        self.add_class("dataset", 1, "with_mask")
        self.add_class("dataset", 2, "without_mask")
        self.add_class("dataset", 3, "mask_weared_incorrect")
        images_dir = dataset_dir + f'/{prefix}images/'
        annotations_dir = dataset_dir + f'/{prefix}annotations/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            xmin = int(box.find('bndbox/xmin').text)
            ymin = int(box.find('bndbox/ymin').text)
            xmax = int(box.find('bndbox/xmax').text)
            ymax = int(box.find('bndbox/ymax').text)
            classtype = box.find('./name').text
            coors = [xmin, ymin, xmax, ymax, classtype]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(box[4]))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']




# test/val set
test_set = MaskDataset()
test_set.load_dataset('masks-dataset\\test', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


# define a configuration for the model
class MasksConfig(Config):
    # define the name of the configuration
    NAME = "mask_cfg"
    EPOCHS = 1
    NUM_CLASSES = 1 + 3
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "mask_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 3
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # BATCH_SIZE = 597

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    Recalls = list()
    Precisions = list()
    for image_id in dataset.image_ids:
        if image_id == 470:
            break
    # for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, precision, recall, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
        # computing bounding boxes for misclassification
        Recalls.append(recall[1])
        Precisions.append(precision[1])
        print(image_id)
    # calculate the mean AP across all images
    mAP = mean(APs)
    mean_recall = mean(Recalls)
    mean_precision = mean(Precisions)
    return mAP, mean_precision, mean_recall

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(25)
        mask, _ = dataset.load_mask(25)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        plt.subplot(n_images, 2, i*2+1)
        # plot raw pixel data
        plt.imshow(image)
        plt.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        plt.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        plt.imshow(image)
        plt.title('Predicted')
        ax = plt.gca()
        # plot each box
        for box in range(len(yhat['rois'])):
            classes_dict = {1:'Wearing Mask',
                        2: 'No mask',
                        3: 'Incorrect Mask Wearing',
                        4: 'person',
                        5: 'person-like',
                        6: 'nonhuman'}
            # get coordinates
            y1, x1, y2, x2 = yhat['rois'][box]
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            if classes_dict[yhat['class_ids'][box]] == 'Wearing Mask':
                rect = Rectangle((x1, y1), width, height, fill=False, color='green')
            else:
                rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            
            # t = plt.text(x1, y1, classes_dict[yhat['class_ids'][box]])
            # t.set_bbox(dict(facecolor='red', alpha=1, edgecolor='red'))
            # draw the box
            ax.add_patch(rect)
    # show the figure
    plt.show()

# plots the image being detected using matplotlib
def detection_image_matplotlib(image, model, cfg):
    # load the image and mask
    # image = dataset.load_image(i)
    # mask, _ = dataset.load_mask(i)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]

    plt.imshow(image)
    plt.title('Predicted')
    ax = plt.gca()
    # plot each box
    for box in range(len(yhat['rois'])):
        # get coordinates
        y1, x1, y2, x2 = yhat['rois'][box]
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        classes_dict = {1:'Wearing Mask',
                    2: 'No mask',
                    3: 'Incorrect Mask Wearing',
                    4: 'person',
                    5: 'person-like',
                    6: 'nonhuman'}
        t = plt.text(x1, y1, classes_dict[yhat['class_ids'][box]])
        t.set_bbox(dict(facecolor='red', alpha=1, edgecolor='red'))
        # draw the box
        ax.add_patch(rect)
        plt.show()

# using open cv to plot the output for object detection
def detection_image_cv2(image, model, cfg):
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]
    if len(yhat['rois']) == 0:
        points = [[0,0]]
    else:
        points = []
    
    for box in range(len(yhat['rois'])):
        # get coordinates
        y1, x1, y2, x2 = yhat['rois'][box]
        points.append([x1, y1+100])
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        if yhat['class_ids'][box] == 1:
            colour = (0, 255, 0)
        else:
            colour = (0, 0, 255)
        image = cv2.rectangle(image, (x2, y2), (x1, y1), colour, 2)
        classes_dict = {1:'Wearing Mask',
                    2: 'No mask',
                    3: 'Incorrect Mask Wearing',
                    4: 'person',
                    5: 'person-like',
                    6: 'nonhuman'}
        
        cv2.putText(image,classes_dict[yhat['class_ids'][box]], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        # points = [[191, 487], [254, 388], [55, 387], [330, 370], [450, 330], [377, 274]]
    perspective_image, added = full_social_distancing(image, points, 100)
    return image, perspective_image, added


###################################################
# # TRAINING this section can be uncommented to train the model. The cfg will need to be edited to configure.
# config = MasksConfig()
# config.display()
# # define the model
# model = MaskRCNN(mode='training', model_dir='./', config=config)
# # load weights (mscoco) and exclude the output layers
# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# # train weights (output layers or 'heads')
# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')
###################################################

# cfg = PredictionConfig()
# print(cfg)
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# model_path = 'anchor ratio double\\mask_rcnn_mask_cfg_0001.h5'
# model.load_weights(model_path, by_name=True)
# data_test = evaluate_model(test_set, model, cfg)
# print(data_test)










# # MAP SCORES
# # create config
# cfg = PredictionConfig()
# cfg.display()
# # define the model
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# # load model weights
# model.load_weights('mask_cfg20210315T1343\\mask_rcnn_mask_cfg_0005.h5', by_name=True)
# # evaluate model on training dataset
# train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# # evaluate model on test dataset
# test_mAP = evaluate_model(test_set, model, cfg)
# print("Test mAP: %.3f" % test_mAP)
# print(os.getcwd())

# # # PREDICTING TEST AND TRAIN IMAGES
# # create config
# cfg = PredictionConfig()
# # define the model
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# # load model weights
# model_path = 'heads+warped\\mask_rcnn_mask_cfg_0025.h5'
# model.load_weights(model_path, by_name=True)

# plot_actual_vs_predicted(train_set, model, cfg, n_images=1)
# # plot predictions for train dataset
# plot_actual_vs_predicted(train_set, model, cfg)
# # # plot predictions for test dataset
# # plot_actual_vs_predicted(test_set, model, cfg)


# # USING WEBCAM
# cfg = PredictionConfig()
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# model_path = 'mask_cfg20210319T1049\\mask_rcnn_mask_cfg_0025.h5'
# model.load_weights(model_path, by_name=True)
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")
# i=0
# while i<1:
#     ret, frame = cap.read()
#     time_1 = time.perf_counter()
#     cv2.imshow('Input', frame)
#     time_2 = time.perf_counter()
#     predicted, perspective_predicted = detection_image_cv2(frame, model, cfg)
#     cv2.imshow('Prediction', predicted)
#     cv2.imshow('Birds eye view', perspective_predicted)
#     i+1
#     fps = 1/(time_2-time_1)
#     print(f'Fps: {fps/1000}')
#     c = cv2.waitKey(1)
#     if c == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()

# using video
# cfg = PredictionConfig()
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# model_path = 'heads + no warped\\mask_rcnn_mask_cfg_0025.h5'
# model.load_weights(model_path, by_name=True)

# capture = cv2.VideoCapture('videos/street2.mp4')
# number_of_frames = 0
# frames = list()
# filenames = list()
# while True:
#     ret, frame = capture.read()
#     if not ret:
#         break
#     frames.append(frame)

# for frame in frames:
#     predicted, bird, added = detection_image_cv2(frame, model, cfg)
#     image_path = os.path.join(os.getcwd(), 'videos\\combined')
#     cv2.imwrite(os.path.join(image_path, f'image_{number_of_frames}.jpg'), added)
#     filenames.append(os.path.join(image_path, f'image_{number_of_frames}.jpg'))
#     height, width, depth = predicted.shape
#     size = (width, height)
#     number_of_frames += 1
#     print(number_of_frames)

# img_array = []
# out = cv2.VideoWriter('videos\combined.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 29.97, size)
# for filename in filenames:
#     img = cv2.imread(filename)
#     img_array.append(img)

# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()

# # using picture
# image = cv2.imread('test_picture_street.jpg')
# cfg = PredictionConfig()
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# model_path = 'mask_cfg20210319T1049\\mask_rcnn_mask_cfg_0025.h5'
# model.load_weights(model_path, by_name=True)
# boxes_image, birds_image = detection_image_cv2(image, model, cfg)
# cv2.imshow('prediction', boxes_image)
# cv2.imshow('birds', birds_image)
# cv2.imwrite('test_street_boxes.jpg', boxes_image)
# cv2.imwrite('test_street_bird.jpg', birds_image)
# cv2.waitKey(0)

# cfg = PredictionConfig()
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# model_path = 'heads + no warped\\mask_rcnn_mask_cfg_0025.h5'
# model.load_weights(model_path, by_name=True)


# data_train = evaluate_model(train_set, model, cfg)
# print(data_train)
# data_test= evaluate_model(test_set, model, cfg)
# print(data_test)

# data = np.concatenate(data_train, data_test)

# df = pd.DataFrame(data, columns=['Train mAP', 'Train Precision', 'Train Recall', 'Test mAP', 'Test Precision', 'Test Recall'])




# cfg = PredictionConfig()
# print(cfg)
# model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# model_path = 'anchor double\\mask_rcnn_mask_cfg_0001.h5'
# model.load_weights(model_path, by_name=True)
# data_test = evaluate_model(test_set, model, cfg)
# print(data_test)