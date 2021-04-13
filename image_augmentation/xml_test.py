from xml.etree.ElementTree import Element, SubElement, tostring, XML
import xml.etree.ElementTree as ET
# from ElementTree_pretty import prettify
from xml.etree import ElementTree
from xml.dom import minidom


# method for creating objects individually
def object_creator(root, label, xmin, ymin, xmax, ymax):
    
    obj = SubElement(root, 'object')
    name = SubElement(obj, 'name')
    name.text = label
    pose = SubElement(obj, 'pose')
    pose.text = 'Unspecified'
    truncated = SubElement(obj, 'truncated')
    truncated.text = '0'
    occluded = SubElement(obj, 'occluded')
    occluded.text = '0'
    difficult = SubElement(obj, 'difficult')
    difficult.text = '0'
    bndbox = SubElement(obj, 'bndbox')
    xminlabel = SubElement(bndbox, 'xmin')
    xminlabel.text = str(xmin)
    yminlabel = SubElement(bndbox, 'ymin')
    yminlabel.text = str(ymin)
    xmaxlabel = SubElement(bndbox, 'xmax')
    xmaxlabel.text = str(xmax)
    ymaxlabel = SubElement(bndbox, 'ymax')
    ymaxlabel.text = str(ymax)
    return obj

# method to create xml file
def file_creator(fdr, image_name, w, h, d):
    annotation = Element('annotation')
    folder = SubElement(annotation, 'folder')
    folder.text = fdr
    filename = SubElement(annotation, 'filename')
    filename.text = image_name
    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(w)
    height = SubElement(size, 'height')
    height.text = str(h)
    depth = SubElement(size, 'depth')
    depth.text = str(d)
    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'
    return annotation


# test code
# annotation = file_creator('s', 'sd', 3, 4, 5)

# object_creator(annotation, 'without_mask', 79, 105, 109, 142)

# data = ET.tostring(annotation, encoding = 'unicode')
# myfile = open('test.xml', 'w')
# myfile.write(data)