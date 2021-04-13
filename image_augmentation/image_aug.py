import imgaug as ia
from xml_test import *

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil
from xml.etree.ElementTree import Element, SubElement, tostring, XML

# converting xml to csv
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[5][0].text),
                        int(member[5][1].text),
                        int(member[5][2].text),
                        int(member[5][3].text)
                        )
                xml_list.append(value)
            except:
                pass
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(xml_list, columns=column_names)
    return df

# function to convert BoundingBoxesOnImage object into DataFrame
def bbs_obj_to_df(bbs_object):
#     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
#     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

# augmenting images

# function to transform images and augment. Adapted from blog post https://medium.com/@a.karazhay/guide-augment-images-and-multiple-bounding-boxes-for-deep-learning-in-4-steps-with-the-notebook-9b263e414dac
def image_aug(df, images_path, aug_images_path, image_prefix, augmentor, annotations_path, aug_annotations_path):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                                ['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax']
                                )
    grouped = df.groupby('filename')
    i=0
    for filename in df['filename'].unique():
    #   get separate data frame grouped by file name
        img_name = aug_images_path+image_prefix+filename
        
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)  
        xml_file = file_creator('aug1_images', img_name, str(group_df['width'][0]), str(group_df['height'][0]) , 3) 
    #   read the image
        image = imageio.imread(images_path+filename)
    #   get bounding boxes coordinates and write into array        
        bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
    #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
    #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
    #   disregard bounding boxes which have fallen out of image pane    
        bbs_aug = bbs_aug.remove_out_of_image()
    #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()
        
    #   don't perform any actions with the image if there are no bounding boxes left in it    
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass
        
    #   otherwise continue
        else:
        #   write augmented image to a file
            imageio.imwrite(aug_images_path+image_prefix+filename, image_aug)  
        #   create a data frame with augmented values of image width and height
            for index, line in group_df.iterrows():
                xmin = line['xmin']
                ymin = line['ymin']
                xmax = line['xmax']
                ymax = line['ymax']
                label = line['class']
                object_creator(xml_file, label, xmin, ymin, xmax, ymax)
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)    
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
        #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix+x)
        #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
        #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
        #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df]) 
        data = ET.tostring(xml_file, encoding = 'unicode')
        xml_path = os.path.join(aug_annotations_path, image_prefix + filename[:-4] + '.xml')
        myfile = open(xml_path, 'w')
        myfile.write(data)     
        print(i)   
        i += 1   

    # return dataframe with updated images and bounding boxes annotations 
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)


    return aug_bbs_xy

# useful augments for masks dataset.
    # gaussian noise for blur as images can be blurry
    # resize and cropping to give more variety
    # colour saturation changes to mimic different environments such as overcast day or bright lights

augmentor = iaa.SomeOf(1, [    
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Affine(rotate=(-60, 60)),
    iaa.Affine(translate_percent={"x":(-0.3, 0.3),"y":(-0.3, 0.3)}),
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.05*255))
    ]
    )

# applying to training data
training_labels_df = xml_to_csv('masks-dataset\\train\\annotations\\')
training_labels_df.to_csv('masks-dataset/training_labels.csv')

test_labels_df = xml_to_csv('masks-dataset\\test\\annotations\\')
test_labels_df.to_csv('masks-dataset/test_labels.csv')


# print(image_aug(
#     training_labels_df, 
#     'masks-dataset\\train\\images\\', 
#     'masks-dataset\\train\\aug1_images\\', 
#     'aug1_', 
#     augmentor,
#     'masks-dataset\\train\\annotations\\',
#     'masks-dataset\\train\\aug1_annotations')
#     )