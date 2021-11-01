import numpy as np
from pycocotools.coco import COCO
import cv2

annFile = '/mnt/data/DATASET/coco/annotations/person_keypoints_train2017.json'
coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

hip_l = 0
hip_r = 0
knee_l = 0
knee_r = 0

for i in imgIds:
    img = coco.loadImgs(i)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    for ann in anns:
        kp = np.array(ann['keypoints'])
        x = kp[0::3]
        y = kp[1::3]
        v = kp[2::3]
        # print(v)
        if v[11] != 0:
            hip_l = hip_l + 1
        if v[12] != 0:
            hip_r = hip_r + 1
        if v[15] != 0:
            knee_l = knee_l + 1
        if v[16] != 0:
            knee_r = knee_r + 1

print(hip_l, hip_r, knee_l, knee_r)
