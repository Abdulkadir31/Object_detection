import os
import glob

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = pb_fname
PATH_TO_CKPT = "C:/Users/abdul/Downloads/object_detection_demo-master/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "C:/Users/abdul/Downloads/object_detection_demo-master/label_map.pbtxt"

# If you want to test the code with your images, just add images files to the PATH_TO_TEST_IMAGES_DIR.
PATH_TO_TEST_IMAGES_DIR =  os.path.join("C:/Users/abdul/Downloads/object_detection_demo-master", "test")

assert os.path.isfile(PATH_TO_CKPT)
assert os.path.isfile(PATH_TO_LABELS)
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
#print(TEST_IMAGE_PATHS)

# %cd /content/models/research/object_detection

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
num_classes = 3

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops


# This is needed to display the images.
# %matplotlib inline


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')



            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def output(path):
    # num = None
    # for image_path in TEST_IMAGE_PATHS:
    
    #     temp_path = image_path.split('\\')
    #     num1 = temp_path[1].split('.')
    #     num = int(num1[0])


    image = Image.open(path)
    im_width, im_height = image.size
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
   
   
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'], 
     	output_dict['detection_classes'],   
     	output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    final_score = output_dict['detection_scores']
    final_score_classes = output_dict['detection_classes']    
    final_detection_boxes = output_dict['detection_boxes']

	# pass_data (accuracy,class,co-ordinates)
    pass_data = []

    count = 0
    total_spalling = 0
    total_cracks = 0
    total_exposed_bricks = 0
    figure = [] 
    for i in range(100):
        if output_dict['detection_scores'] is None or final_score[i] > 0.5:

            pass_data.append(final_score_classes[i])
            pass_data.append(final_score[i])
            pass_data.append(final_detection_boxes[i])
            
            count = count + 1
            figure.append(final_detection_boxes[i])
            if final_score_classes[i] == 3:
            	total_spalling += 1
            elif final_score_classes[i] == 2:
            	total_exposed_bricks += 1
            else:
            	total_cracks += 1 

    print("Number of boxes:=", count)
    print("Spalling Count:=",total_spalling)
    print("Exposed Bricks Count:=",total_exposed_bricks)
    print("Cracks Count:=",total_cracks)
    
   
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.savefig('static/img1.png')

        

        # image2 = cv2.imread(image_path)

        # count = 1
        # for single_figure in figure:

    	   #  left, right, top, bottom = (single_figure[1] * im_width, single_figure[3] * im_width, single_figure[0] * im_height, single_figure[2] * im_height)
    	    

    	   #  crop = image2[int(top)-10:int(bottom)+10, int(left)-10:int(right)+10]
    	   #  cv2.imwrite('C:/Damage_detection_1/boxes/'+str(num)+'_'+str(count)+'.png', crop)
    	   #  count += 1

       # Create Files for accuracy
        # i = 0
       
        # with open('C:/Damage_detection_1/accuracy_files/'+str(num)+'.txt','w') as f:
        #     while i < len(pass_data):
        #         data = ''
        #         if pass_data[i] == 2:
        #             data += 'Exposed_Bricks '
        #         elif pass_data[i] == 3:
        #             data += 'Spalling '
        #         else:
        #             data += 'Crack ' 
        #         data += str(pass_data[i+1])+' '

        #         left, right, top, bottom = (pass_data[i+2][1] * im_width, pass_data[i+2][3] * im_width, pass_data[i+2][0] * im_height, pass_data[i+2][2] * im_height)

        #         data += str(int(left))+' '+str(int(top))+' '+str(int(right-left))+' '+str(int(bottom-top))+'\n'

        #         f.write(data)
        #         i += 3
        
    return str(count)+" "+str(total_spalling)+" "+str(total_exposed_bricks)+" "+str(total_cracks)
    

