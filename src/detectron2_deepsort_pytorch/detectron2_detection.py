from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

MODEL = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

class Detectron2:

    def __init__(self, detectron2_checkpoint=None, use_cuda=True):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(MODEL))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
        if not use_cuda: self.cfg.MODEL.DEVICE='cpu'
        self.predictor = DefaultPredictor(self.cfg)

    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax

    def detect(self, im):
        outputs = self.predictor(im)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        bbox_xcycwh, cls_conf, cls_ids = [], [], []

        for (box, _class, score) in zip(boxes, classes, scores):
            x0, y0, x1, y1 = box
            bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
            cls_conf.append(score)
            cls_ids.append(_class)

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids)
