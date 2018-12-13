# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import json
import botocore.session

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

class Detector:
	def detectObject(self, imName):	
		# Name of the directory containing the object detection module we're using
		MODEL_NAME = 'inference_graph'
		IMAGE_NAME = 'test_knife.jpg'

		# Grab path to current working directory
		CWD_PATH = os.getcwd()

		# Path to frozen detection graph .pb file, which contains the model that is used
		# for object detection.
		PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

		# Path to label map file
		PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labelmap.pbtxt')

		# Path to image
		PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

		# Number of classes the object detector can identify
		NUM_CLASSES = 2

		# Load the label map.
		# Label maps map indices to category names, so that when our convolution
		# network predicts `5`, we know that this corresponds to `king`.
		# Here we use internal utility functions, but anything that returns a
		# dictionary mapping integers to appropriate string labels would be fine
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)

		# Load the Tensorflow model into memory.
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

			sess = tf.Session(graph=detection_graph)

		# Define input and output tensors (i.e. data) for the object detection classifier

		# Input tensor is the image
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

		# Output tensors are the detection boxes, scores, and classes
		# Each box represents a part of the image where a particular object was detected
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

		# Each score represents level of confidence for each of the objects.
		# The score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

		# Number of objects detected
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		# Load image using OpenCV and
		# expand image dimensions to have shape: [1, None, None, 3]
		# i.e. a single-column array, where each item in the column has the pixel RGB value
		#image = cv2.imread(PATH_TO_IMAGE)
		image = imName
		#image = cv2.imread(img)
		image_expanded = np.expand_dims(imName, axis=0)

		#cv2.imshow('Object detector', image_expanded)
		
		# Perform the actual detection by running the model with the image as input
		(boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: image_expanded})

		#print('TEST')
		#print(num)
		#print(classes.shape)
		#print(scores.shape)
		#print(boxes.shape)

		thresh = 0.1

		# Draw the results of the detection (aka 'visulaize the results')

		vis_util.visualize_boxes_and_labels_on_image_array(
			image,
			np.squeeze(boxes),
			np.squeeze(classes).astype(np.int32),
			np.squeeze(scores),
			category_index,
			use_normalized_coordinates=True,
			line_thickness=4,
			min_score_thresh=thresh)
		
		img = cv2.imencode('.jpg', image)[1].tobytes()
		
		sq_score = np.squeeze(scores)
		sq_classes = np.squeeze(classes).astype(np.int32)

		#print(sq_score[sq_score > thresh])

		detected_class = sq_classes[np.nonzero(sq_score > thresh)]

		class_freq = np.unique(detected_class, return_counts=True)
		
		#with open('G:\Pramod\Cogintive\Project\Trigger\config.json','r') as f:
		#	config_data = f.read()

		#config = json.loads(config_data)

		#access_key = config['AWS Access Keys']
		#secret_access_key = config['AWS Secret Keys']

		session = botocore.session.get_session()

		sns = session.create_client('sns', region_name="us-east-1")

		email_to = "nagare.p@husky.neu.edu"
		email_from = "CCTV Footage"
		Message = email_to+"|"+email_from
		
		if class_freq[0].size > 0 :
			print("Email Sent")
			sns.publish(TopicArn="arn:aws:sns:us-east-1:515649345368:cognitive",Message=Message)
			
		return img
		
		#return image
	
# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
