from detectron2.config import CfgNode as CN


def add_vertex_detection_config(cfg):
    """Add config to vertex detection"""
    cfg.MODEL.VERTEX_HEAD = CN()
    cfg.MODEL.VERTEX_HEAD.NAME = "VertexHead"
    cfg.MODEL.VERTEX_HEAD.IN_FEATURES = ('p2',)  # ('p2','p3','p4','p5')
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.TRAIN = ("plane_train",)
    cfg.DATASETS.TEST = ()