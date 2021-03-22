import os
import sys
import glob
import json
import cv2
import numpy as np
from pycocotools import mask
import warnings

#
# from detectron2.data.catalog import DatasetCatalog, MetadataCatalog, Metadata
from detectron2.structures import BoxMode


H, W = 512, 512
h, w = 128, 128
_plane_area = 8 * 8
margin = 2 * (H / h)

sys.path.append(".")
sys.path.append("..")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.uint32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

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
    return [np.clip(np.min(idx[:, 1]), 0, h-1e-4), np.clip(np.min(idx[:, 0]), 0, w-1e-4),
            np.clip(np.max(idx[:, 1]), 0, h-1e-4), np.clip(np.max(idx[:, 0]), 0, w-1e-4)]
    # YXYX for ours' format
    # return [np.clip(np.min(idx[:, 0]) - margin, 0, W-1e-4), np.clip(np.min(idx[:, 1]) - margin, 0, H-1e-4),
    #         np.clip(np.max(idx[:, 0]) + margin, 0, W-1e-4), np.clip(np.max(idx[:, 1]) + margin, 0, H-1e-4)]


def coco_format(filename, idx):
    record = {}
    record["file_name"] = os.path.split(filename)[-1].replace("_a0_label.npz", ".png")
    record["image_id"] = idx
    record["height"] = H
    record["width"] = W
    with np.load(filename) as junc_npz:
        junc = junc_npz["junc"]  # [Na, 3] [0~128]
    with open(filename.replace("a0_label.npz", "label.json"), "r") as f:
        data = json.load(f)
        tris2plane = np.array(data["tris2plane"])
    with np.load(filename.replace("a0_label.npz", "GL.npz")) as plane_npz:
        idmap = plane_npz["idmap_face"]
        idmap = np.array(tris2plane[idmap.ravel() - 1]).reshape(idmap.shape)
        # the background id from 0 to max after tris2plane !!!
        idmap = downsample_nearest(idmap, new_size=(h, w))

    unval, _ = np.unique(idmap, return_inverse=True)

    _junc_ = junc
    # _junc_[:, :2] = _junc_[:, :2] * (H / h)  # 映射到 [0-512]
    for i, j in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if idmap[i, j] != unval[-1]:  # if not background
            _junc_ = np.vstack((_junc_, np.array([i, j, 0])))
    jmap = np.zeros((h, w))
    for v in _junc_:
        vint = to_int(v[:2])
        jmap[vint] = 1
    encode_jmap = np.asarray(jmap,dtype=np.uint8,order="F")
    gt_jmap = mask.encode(encode_jmap)

    objs = []
    for k, val in enumerate(unval[:-1]):
        gt_mask = np.asarray(idmap == val, order="F")
        encoded_gt = mask.encode(gt_mask)
        gt_bbox = toBbox(gt_mask)
        area_gt = np.sum(gt_mask)
        if area_gt < _plane_area:
            continue
        obj = {
            "id": k,
            "bbox": gt_bbox,
            "bbox_mode": BoxMode.XYXY_ABS, # wait for confirm
            "segmentation": encoded_gt,
            "category_id": 0,
        }
        objs.append(obj)
    record["annotations"] = objs
    record["junc_heat_map"] = gt_jmap
    return record


def get_plane_dicts(root,stage="train"):
    # filelist = sorted(glob.glob(f"{root}/[0-9]*/*_a0_label.npz"))
    filelist =sorted(glob.glob(f"{root}/000/*_a0_label.npz"))
    dataset_dicts = []
    print("filelist length is %s" %len(filelist))
    for idx in range(len(filelist)):
        single_record = coco_format(filelist[idx],idx)
        dataset_dicts.append(single_record)
        print(idx)
    os.makedirs(output_folder, exist_ok=True)
    cache_path = os.path.join(output_folder, f"{dataset_name}_{stage}_coco_format.json")
    with open(cache_path, "w") as json_file:
        json.dump(dataset_dicts, json_file, cls=MyEncoder)
    return dataset_dicts


if __name__ == "__main__":
    root_dir = "/home/zll/HoliCity-MaskRCNN/dataset/r2"
    output_folder = "/home/zll/HoliCity-MaskRCNN/output"
    dataset_name = "plane_net"  # this name just for test,need to confirm and change to the right name
    dataset_dicts = get_plane_dicts(f"{root_dir}/train", stage="train")
    print("sucess!")