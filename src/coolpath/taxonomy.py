from __future__ import annotations
import numpy as np

CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
],
                              dtype=np.uint8)
CITYSCAPES_NAME_TO_ID = {name: i for i, name in enumerate(CITYSCAPES_CLASSES)}

BFMS_ID2LABEL = {
    0: "Background",
    1: "Wood/Bamboo",
    2: "Ground tile",
    3: "Brick",
    4: "Cardboard/Paper",
    5: "Tree",
    6: "Roof tile",
    7: "Ceramic",
    8: "Chalkboard/Blackboard",
    9: "Asphalt",
    10: "Cement/Concrete",
    11: "Composite decorative board",
    12: "Rammed earth",
    13: "Fabric/Cloth",
    14: "Water",
    15: "Windows with metal fences",
    16: "Foliage",
    17: "Food",
    18: "Fur",
    19: "Pottery",
    20: "Glass",
    21: "Hair",
    22: "Roofing waterproof material",
    23: "Ice",
    24: "Leather",
    25: "Carved brick",
    26: "Metal",
    27: "Mirror",
    28: "Enamel",
    29: "Paint/Coating/Plaster",
    30: "Window screen",
    31: "Whiteboard",
    32: "Photograph/Painting/Airbrushed fabric",
    33: "Plastic, clear",
    34: "Plastic, non-clear",
    35: "Rubber/Latex",
    36: "Sand",
    37: "Skin/Lips",
    38: "Sky",
    39: "Snow",
    40: "Engineered Stone/Imitation Stone",
    41: "Soil/Mud",
    42: "Natural Stone",
}
BFMS_CLASSES = [BFMS_ID2LABEL[i] for i in range(43)]
BFMS_NAME_TO_ID = {name: i for i, name in enumerate(BFMS_CLASSES)}

SEM_ROAD = 0
SEM_SIDEWALK = 1
SEM_BUILDING = 2
SEM_VEGETATION = 8
SEM_TERRAIN = 9
SEM_SKY = 10
IGNORE_INDEX = 255


def colorize_cityscapes(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask != IGNORE_INDEX) & (mask < len(CITYSCAPES_PALETTE))
    out[valid] = CITYSCAPES_PALETTE[mask[valid]]
    return out
