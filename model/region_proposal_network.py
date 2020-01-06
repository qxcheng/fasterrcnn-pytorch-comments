import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    # ratios: width/height
    # anchor_scales: areas of anchors
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16, proposal_creator_params=dict(),):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios) # (9, 4)
        self.feat_stride = feat_stride                                                      # 16
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]                                                # 9

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        # img_size: [600,800]
        # scale: float=1.6=600/375=800/500
        n, _, hh, ww = x.shape                                                                   # [1, 512, 37, 50]
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww) # (16650, 4)    返回值

        n_anchor = anchor.shape[0] // (hh * ww)                                                  # int=9
        h = F.relu(self.conv1(x))                                                                # [1, 512, 37, 50]

        rpn_locs = self.loc(h)                                                                   # [1, 36, 37, 50]
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)                      # [1,16650,4]   返回值
        rpn_scores = self.score(h)                                                               # [1,18,37,50]
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()                                 # [1,37,50,18]
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)           # [1,37,50,9,2]
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()                           # [1,37,50,9]
        rpn_fg_scores = rpn_fg_scores.view(n, -1)                                                # [1,16650]
        rpn_scores = rpn_scores.view(n, -1, 2)                                                   # [1,16650,2]   返回值

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),       # [16650,4]
                rpn_fg_scores[i].cpu().data.numpy(),  # [16650]
                anchor,                               # (16650, 4)
                img_size,                             # [600,800]
                scale=scale)                          # float=1.6
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)                      # (300,4)
        roi_indices = np.concatenate(roi_indices, axis=0)        # (300,)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

# 产生anchor 为原图尺寸
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # params: (9, 4)  16  60  40
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)     # (60,)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)      # (40,)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)              # (60, 40) (60, 40)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)  # (2400, 4)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))        # (2400, 9, 4)
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)        # (21600, 4)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
