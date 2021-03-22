import os
import sys
import glob
import json
import cv2
import numpy as np
import warnings

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog, Metadata
from detectron2.structures import BoxMode


H, W = 512, 512
h, w = 128, 128
margin = 2 * (H / h)

sys.path.append(".")
sys.path.append("..")
rootdir = "/home/zll/HoliCity-MaskRCNN/dataset/r2/"

def to_int(x):
    return tuple(map(int, x))


def downsample_nearest(arr, new_size=(128, 128)):
    x = np.arange(0, arr.shape[0], int(arr.shape[0]/new_size[0]))
    y = np.arange(0, arr.shape[1], int(arr.shape[1]/new_size[1]))
    assert len(x) == new_size[0] and len(y) == new_size[1]
    x_, y_ = np.meshgrid(x, y)

    new_arr = arr[y_.flatten(), x_.flatten()]
    return new_arr.reshape(new_size)


def toBbox(mask_array):
    idx = np.argwhere(mask_array == 1)
    # XYXY for detectron2 default format
    return [np.clip(np.min(idx[:, 1]) - margin, 0, H-1e-4), np.clip(np.min(idx[:, 0]) - margin, 0, W-1e-4),
            np.clip(np.max(idx[:, 1]) + margin, 0, H-1e-4), np.clip(np.max(idx[:, 0]) + margin, 0, W-1e-4)]
    # YXYX for ours' format
    # return [np.clip(np.min(idx[:, 0]) - margin, 0, W-1e-4), np.clip(np.min(idx[:, 1]) - margin, 0, H-1e-4),
    #         np.clip(np.max(idx[:, 0]) + margin, 0, W-1e-4), np.clip(np.max(idx[:, 1]) + margin, 0, H-1e-4)]


def process_item(filename,idx):
    height, width = cv2.imread(filename).shape[:2]
    record = {}
    record["file_name"] = os.path.split(filename)[-1].replace("_a0_label.npz", ".png")
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width
    with np.load(filename) as junc_npz:
        junc = junc_npz["junc"]  # [Na, 3] [0~128]
    with open(filename.replace("a0_label.npz", "label.json"), "r") as f:
        data = json.load(f)
        tris2plane = np.array(data["tris2plane"])
    with np.load(filename.replace("a0_label.npz", "GL.npz")) as plane_npz:
        idmap = plane_npz["idmap_face"]
        idmap = np.array(tris2plane[idmap.ravel() - 1]).reshape(idmap.shape)
        # the background id from 0 to max after tris2plane !!!
        idmap = downsample_nearest(idmap, new_size=(H, W))

    unval, _ = np.unique(idmap, return_inverse=True)

    _junc_ = junc
    _junc_[:, :2] = _junc_[:, :2] * (H / h)  # 映射到 [0-512]
    for i, j in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
        if idmap[i, j] != unval[-1]:  # if not background
            _junc_ = np.vstack((_junc_, np.array([i, j, 0])))
    _junc_[:, 0] = np.clip(_junc_[:, 0], 0, H - 1e-4)
    _junc_[:, 1] = np.clip(_junc_[:, 1], 0, W - 1e-4)

    jmap = np.zeros((h, w))
    _junc_[:, :2] = np.clip(_junc_[:, :2] / (H / h), 0, h - 1e-4)
    for v in _junc_:
        vint = to_int(v[:2])
        jmap[vint] = 1

    objs = []
    for k, val in enumerate(unval[:-1]):
        gt_mask = np.asarray(idmap == val, order="F")
        gt_bbox = toBbox(gt_mask)
        obj = {
            "bbox": gt_bbox,
            "bbox_mode": BoxMode.XYXY_ABS, # wait for confirm
            "segmentation": gt_mask,
            "category_id": 0,
        }
        objs.append(obj)
    record["annotations"] = objs
    record["junc_heat_map"] = jmap
    return record


def get_plane_dicts(root):
    filelist = sorted(glob.glob(f"{root}/[0-9]*/*_0_a0_label.npz"))
    dataset_dicts = []
    print("filelist length is%s"%len(filelist))
    for idx in range(len(filelist)):
        single_record = process_item(filelist[idx],idx)
        dataset_dicts.append(single_record)


for d in ["train", "val"]:
    DatasetCatalog.register("plane_" + d, lambda d=d: get_plane_dicts(rootdir + d))
    MetadataCatalog.get("plane_" + d).set(thing_classes=["plane"])
plane_metadata = MetadataCatalog.get("plane_train")