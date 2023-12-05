import os
import random
import json
import cv2
import numpy as np
from scipy import ndimage

import mrcnn.model as modellib
import mrcnn.visualize as visualize
import mrcnn.config as config

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
WEIGHT_PATH = os.path.join(ROOT_DIR, "mrcnn/mask_rcnn_coco.h5")

# Directory of videos to run detection on
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Directory of outputs
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

TEST_NAME = 'test2'

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class SimpleConfig(config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

def unpack_rcnn_data(r, j):
    object_name = CLASS_NAMES[r['class_ids'][j]]
    
    mask = r['masks'][:, :, j].astype(int)
    commask = np.flip(np.array(ndimage.measurements.center_of_mass(mask))).astype(int).tolist()
    
    ground_x = commask[0]
    ground_y = np.max(np.argwhere(mask == 1)[:,0])
    ground_mask = np.array([ground_x, ground_y]).astype(int).tolist()

    box = r['rois'][j]
    x1 = box[1]
    x2 = box[3]
    y1 = box[0]
    y2 = box[2]
    combox = np.array([(x1 + x2) / 2, (y1 + y2) / 2]).astype(int).tolist()

    h = int(y2 - y1)
    w = int(x2 - x1)

    score = float(r['scores'][j]) # not good enough

    return commask, combox, ground_mask, object_name, score, h, w

def detect_video(model, video_path):
    info = {}
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            results = model.detect([frame], verbose=1)
            r = results[0]
            # mask_image is my new function in mrcnn.visualize
            masked_frame = visualize.mask_image(frame, r['rois'], r['masks'], r['class_ids'], 
                                                CLASS_NAMES, r['scores'])

            info[frame_id] = []
            for j in range(len(r['class_ids'])):          
                commask, combox, ground_mask, object_name, score, h, w = unpack_rcnn_data(r, j)
                object_info = {'mask_com': commask, 'box_center': combox, 'ground_point': ground_mask, 'class': object_name, 'score': score, 'height': h, 'width': w}
                info[frame_id].append(object_info)

            # Display the resulting frame
            cv2.imshow('Frame', masked_frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
        # Break the loop
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return info

def detect_image(model):
    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                CLASS_NAMES, r['scores'])

if __name__ == "__main__":
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=SimpleConfig())

    # Load weights trained on MS-COCO
    model.load_weights(WEIGHT_PATH, by_name=True)
    #detect_image(model)
    video_path = os.path.join(VIDEO_DIR, TEST_NAME + '.mp4')
    info = detect_video(model, video_path)
    with open(os.path.join(OUTPUT_DIR, TEST_NAME + '.json'), 'w') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)