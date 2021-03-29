# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
import matplotlib.pyplot as plt
from mrcnn.config import Config
from matplotlib.patches import Rectangle
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class BloodDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "WBC")
		self.add_class("dataset", 2, "RBC")
		self.add_class("dataset", 3, "Platelets")
		# define data locations
		images_dir = dataset_dir + '/Images/'
		annotations_dir = dataset_dir + '/Annotations/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			print(img_path,ann_path)
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
		for box in root.iter('object'):
			# print(box)
			ymin, xmin, ymax, xmax = None, None, None, None
			name =box.find('name').text
			ymin = int(box.find("bndbox/ymin").text)
			xmin = int(box.find("bndbox/xmin").text)
			ymax = int(box.find("bndbox/ymax").text)
			xmax = int(box.find("bndbox/xmax").text)
			coors = [xmin, ymin, xmax, ymax,name]
	
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

			# print(box)	
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index(box[4]))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

class BloodConfig(Config):
	# Give the configuration a recognizable name
	NAME = "blood_cfg"
	# Number of classes 
	NUM_CLASSES = 4
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 131

# prepare config
config = BloodConfig()
config.display()
# train set
train_set = BloodDataset()
train_set.load_dataset(dir_path+'/Training', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
test_set = BloodDataset()
test_set.load_dataset(dir_path+'/Testing', is_train=False)
test_set.prepare()
model = MaskRCNN(mode='training', model_dir='./', config=config)

model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
#train
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')


class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "blood_cfg"
	# number of classes 
	NUM_CLASSES = 4
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir=dir_path, config=cfg)

def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
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
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP
#folder name must be changed accordingly for the output folder 
model.load_weights(dir_path+'\\blood_cfg20210327T0816\mask_rcnn_blood_cfg_0010.h5', by_name=True)

# evaluate model on train dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)


def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# print(yhat.rois)
		# define subplot
		plt.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		plt.imshow(image)
		plt.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			plt.imshow(mask[:, :, j])
		# get the context for drawing boxes
		plt.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		plt.imshow(image)
		plt.title('Predicted')
		ax = plt.gca()
		# plot each box
		for box in yhat['rois']:
			print(box)
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	plt.show()
#plot prediction vs actual
plot_actual_vs_predicted(train_set, model, cfg)
plot_actual_vs_predicted(test_set, model, cfg)