import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from scipy import ndimage

from detectron2_deepsort_pytorch.deep_sort import DeepSort
import re

ROOT_DIR = os.path.abspath("./")

VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "datastream/detectron2_deepsort/tracked")
TEST_NAME = "05_23_sync"
CAM = 0

MODEL = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

#unpack the model output data
def unpack_detectron_data(data, idx, class_mapper):
    classes = data.pred_classes
    boxes = data.pred_boxes
    masks = data.pred_masks
    scores = data.scores

    class_num = classes[idx].cpu().detach().item()

    mask = masks[idx].cpu().detach().numpy().astype(int)
    commask = np.flip(np.array(ndimage.measurements.center_of_mass(mask))).astype(int).tolist()

    ground_x = commask[0]
    ground_y = np.max(np.argwhere(mask == 1)[:,0])
    ground_mask = np.array([ground_x, ground_y]).astype(int).tolist()

    box = boxes[idx].tensor.cpu().detach().numpy()[0]
    x1 = box[0]
    x2 = box[2]
    y1 = box[1]
    y2 = box[3]
    combox = np.array([(x1 + x2) / 2, (y1 + y2) / 2]).astype(int).tolist()

    h = int(y2 - y1)
    w = int(x2 - x1)

    score = float(scores[idx].cpu().detach().item())
    object_name = class_mapper[class_num]

    return commask, combox, ground_mask, object_name, score, h, w

#detect on video
def detect_video(predictor, video_path, class_mapper):
    info = {}
    info2 = {}
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame_id += 1
            results = predictor(frame)
            r = results["instances"]

            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            #out = v.draw_instance_predictions(results["instances"].to("cpu"))

            bbox_xcycwh, cls_conf, ob_names = [], [], []
            info[frame_id] = []
            info2[frame_id] = []
            for j in range(len(r)):
                commask, combox, ground_mask, object_name, score, h, w = unpack_detectron_data(r, j, class_mapper)
                bbox_xcycwh.append([combox[0], combox[1], w, h])
                cls_conf.append(score)
                ob_names.append(object_name)
                object_info = {'mask_com': commask, 'box_center': combox, 'ground_point': ground_mask, 'class': object_name, 'score': score, 'height': h, 'width': w}
                info[frame_id].append(object_info)

            bbox_xcycwh = np.array(bbox_xcycwh, dtype=np.int)

            #output: xyxy
            outputs = deepsort.update(bbox_xcycwh, np.array(cls_conf), frame)
            if len(outputs) > 0:
                track_ids = outputs[:, 4]
                bbox_xyxy = outputs[:, :4]
                bbox_xcycwh = []
                for a in bbox_xyxy:
                    bbox_xcycwh.append(xyxy_to_xywh(a))
                bbox_xcycwh = np.array(bbox_xcycwh, dtype=np.int)

            if frame_id > 2:
                i = 0
                for j in bbox_xcycwh:
                    combox = np.array([j[0],j[1]]).astype(int).tolist()
                    commask = combox
                    ground_point = np.array([combox[0],combox[1]+j[3]/2]).astype(int).tolist()
                    object_name = ob_names[i]
                    score = float(cls_conf[i])
                    h = int(j[3])
                    w = int(j[2])
                    object_info = {'cam_id': CAM, 'id': str(track_ids[i]), 'mask_com': commask, 'ground_point': ground_point, 'class': object_name, 'score': score, 'height': h, 'width': w}
                    info2[frame_id].append(object_info)
                    i+=1
                    if len(bbox_xcycwh) > len(ob_names):
                        #print(len(ob_names))
                        if (i+1) > len(ob_names):
                            break

            # Display the resulting frame
            #cv2.imshow('pic', out.get_image()[:, :, ::-1])

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return info2

def xyxy_to_xywh(xyxy):
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    return np.array([int(x_temp), int(y_temp), int(w_temp), int(h_temp)])

def detect_all_videos(predictor, filenames, class_mapper):
    output_path = os.path.join(OUTPUT_DIR, TEST_NAME + '_det')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for filename in filenames:
        CAM = re.search(r'\d+', filename).group(0)
        video_path = os.path.join(os.path.join(VIDEO_DIR, TEST_NAME), filename)
        name = filename.split('.')[0] + ".json"
        print(os.path.join(OUTPUT_DIR, name))
        info = detect_video(predictor, video_path, class_mapper)
        with open(os.path.join(output_path, name), 'w') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)

    class_mapper = {}
    for i, name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes):
        class_mapper[i] = name

    predictor = DefaultPredictor(cfg)

    deepsort = DeepSort("detectron2_deepsort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", use_cuda=True)

    _, _, filenames = next(os.walk(os.path.join(VIDEO_DIR, TEST_NAME)))

    detect_all_videos(predictor, filenames, class_mapper)