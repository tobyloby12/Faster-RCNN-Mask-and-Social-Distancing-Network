from network_again import *
from mrcnn.model import MaskRCNN

# train set
train_set = MaskDataset()
train_set.load_dataset('masks-dataset\\train', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/val set
test_set = MaskDataset()
test_set.load_dataset('masks-dataset\\test', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


# TRAINING
config = MasksConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=50, layers='heads')