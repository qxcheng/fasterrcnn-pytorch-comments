import numpy as np
import numpy as xp

import six                  # six.moves 直接屏蔽python2 python3的差异
from six import __init__

# 根据偏差修正bbox
def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.
    g_y = p_h * d_y + p_y
    g_x = p_w * d_x + p_x
    g_h = p_h * exp(d_h)
    g_w = p_w * exp(d_w)

    Args:
        src_bbox (array): (R, 4)
        loc (array): (R, 4), (t_y, t_x, t_h, t_w).
    Return:
        array
    """
    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]   #源框高度
    src_width = src_bbox[:, 3] - src_bbox[:, 1]    #源框宽度
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height  #中心点y坐标
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width   #中心点x坐标

    dy = loc[:, 0::4]  #修正量
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]

    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox

# 根据两个bbox算偏差
def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    Args:
        src_bbox (array): (R, 4)
        dst_bbox (array): (R, 4)
    Returns:
        array: (R, 4), (t_y, t_x, t_h, t_w).
    """
    height = src_bbox[:, 2] - src_bbox[:, 0]  #源框高度
    width = src_bbox[:, 3] - src_bbox[:, 1]   #源框宽度
    ctr_y = src_bbox[:, 0] + 0.5 * height     #中心点y坐标
    ctr_x = src_bbox[:, 1] + 0.5 * width      #中心点x坐标

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]    #目标框高度
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]     #目标框宽度
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height  #中心点y坐标
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width   #中心点x坐标

    eps = xp.finfo(height.dtype).eps  #最小的正数
    height = xp.maximum(height, eps)  #保证非负
    width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    Args:
        bbox_a (array): (N, 4), numpy.float32
        bbox_b (array): (K, 4), numpy.float32
    Returns:
        array: (N, K)
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.
    Args:
        base_size (number): The width and the height of the reference window.
        ratios (list of floats): This is ratios of width to height of the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
    Returns:
        ~numpy.ndarray:
        An array of shape : (R, 4), bbox(9, 4)
    """
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.

    return anchor_base                              # (9, 4)
'''
[[ -37.254833  -82.50967    53.254833   98.50967 ]
 [ -82.50967  -173.01933    98.50967   189.01933 ]
 [-173.01933  -354.03867   189.01933   370.03867 ]
 [ -56.        -56.         72.         72.      ]
 [-120.       -120.        136.        136.      ]
 [-248.       -248.        264.        264.      ]
 [ -82.50967   -37.254833   98.50967    53.254833]
 [-173.01933   -82.50967   189.01933    98.50967 ]
 [-354.03867  -173.01933   370.03867   189.01933 ]]
'''

if __name__ == '__main__':
    anchor_base = generate_anchor_base()
    print(anchor_base, anchor_base.shape)