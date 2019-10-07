#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from peleenet import build_net
from data import BaseTransform, VOC_CLASSES
from utils.core import *
from utils.pycocotools.coco import COCO


parser = argparse.ArgumentParser(description='Pelee Testing')
parser.add_argument('-c', '--config', default='configs/Pelee_VOC.py')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-m', '--trained_model', default='./weights/Pelee_VOC.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-t', '--thresh', default=0.5, type=float,
                    help='visidutation threshold')
parser.add_argument('--show', action='store_true',
                    help='Whether to display the images')
args = parser.parse_args()

print_info(' ----------------------------------------------------------------------\n'
           '|                       Pelee Demo Program                              |\n'
           ' ----------------------------------------------------------------------', ['yellow', 'bold'])

global cfg
cfg = Config.fromfile(args.config)
anchor_config = anchors(cfg.model)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)
net = build_net('test', cfg.model.input_size, cfg.model)
init_net(net, cfg, args.trained_model)
print_info('===> Finished constructing and loading model', ['yellow', 'bold'])
net.eval()

num_classes = cfg.model.num_classes

imgs_path_dict = {'VOC': 'imgs/VOC', 'COCO': 'imgs/COCO'}
im_path = imgs_path_dict['VOC']


imgs_result_path = os.path.join(im_path, 'result')
if not os.path.exists(imgs_result_path):
    os.makedirs(imgs_result_path)

with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        net = net.cuda()
        priors = priors.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
_preprocess = BaseTransform(
    cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
detector = Detect(num_classes,
                  cfg.loss.bkg_label, anchor_config)


def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base)
          for x in range(num_classes)]
cats = [_.strip().split(',')[-1]
        for _ in open('data/coco_labels.txt', 'r').readlines()]
label_config = {'VOC': VOC_CLASSES, 'COCO': tuple(['__background__'] + cats)}
labels = label_config['VOC']


def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cX = int((box[0] + box[2]) / 2.0)
        cY = int((box[1] + box[3]) / 2.0)
        centroids = (cX, cY)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)
        cv2.line(imgcv, centroids, centroids, (0,0,255), 7)
        print(labels[cls_indx]," C-coordinate: ", centroids)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15),
                        0, 2e-3 * h, (255, 255, 255), thick // 2)

    return imgcv

cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    loop_start = time.time()
    cv2.imshow('ori', image)
    w, h = image.shape[1], image.shape[0]
    img = _preprocess(image).unsqueeze(0)
    if cfg.test_cfg.cuda:
        img = img.cuda()
    scale = torch.Tensor([w, h, w, h])
    out = net(img)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0] * scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    allboxes = []
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        soft_nms = cfg.test_cfg.soft_nms
        # min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
        keep = keep[:cfg.test_cfg.keep_per_class]
        c_dets = c_dets[keep, :]
        allboxes.extend([_.tolist() + [j] for _ in c_dets])

    allboxes = np.array(allboxes)
    boxes = allboxes[:, :4]
    scores = allboxes[:, 4]
    cls_inds = allboxes[:, 5]
    im2show = draw_detection(image, boxes, scores, cls_inds, -1, args.thresh)
    if im2show.shape[0] > 1100:
        im2show = cv2.resize(im2show,
                             (int(2000. * float(im2show.shape[1]) / im2show.shape[0]), 2000))
    if args.show:
        cv2.imshow('result', im2show)
        #print("Time for processing one pic: ",time.time() - loop_start)
    
    #save images
    #filename = os.path.join(imgs_result_path, '{}_stdn.jpg'.format(os.path.basename(str(count))))
    #bucv2.imwrite(filename, im2show)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
