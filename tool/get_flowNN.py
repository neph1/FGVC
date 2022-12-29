from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import copy
import numpy as np
import scipy.io as sio
from utils.common_utils import interp, BFconsistCheck, \
    FBconsistCheck, consistCheck, get_KeySourceFrame_flowNN

import concurrent.futures
from itertools import repeat

def calculateMask(indFrame, mask_tofill, imgW, imgH, num_candidate, HaveFlowNN, consistencyMap, videoNonLocalFlowF, videoNonLocalFlowB, mask, video, videoBN, videoFN, sub, args):
    HaveNN, NotHaveNN, HaveNN_sum, imgKeySourceFrameFlowNN = calculateHaveNN(indFrame, imgW, imgH, num_candidate, HaveFlowNN, consistencyMap, videoNonLocalFlowF, videoNonLocalFlowB, mask, video, sub, args.Nonlocal, args.consistencyThres)
    res, video = calculatePixel(indFrame, videoBN, videoFN, imgW, imgH, num_candidate, consistencyMap, imgKeySourceFrameFlowNN, HaveNN, NotHaveNN, HaveNN_sum, mask, video, args.Nonlocal, args.alpha)
    
    return res, video


def calculateHaveNN(indFrame, imgW, imgH, num_candidate, HaveFlowNN, consistencyMap, videoNonLocalFlowF, videoNonLocalFlowB, mask, video, sub, nonLocal, consistencyThres):
    if nonLocal:
        consistencyMap[:, :, 2, indFrame], _ = consistCheck(
            videoNonLocalFlowB[:, :, :, 0, indFrame],
            videoNonLocalFlowF[:, :, :, 0, indFrame])
        consistencyMap[:, :, 3, indFrame], _ = consistCheck(
            videoNonLocalFlowB[:, :, :, 1, indFrame],
            videoNonLocalFlowF[:, :, :, 1, indFrame])
        consistencyMap[:, :, 4, indFrame], _ = consistCheck(
            videoNonLocalFlowB[:, :, :, 2, indFrame],
            videoNonLocalFlowF[:, :, :, 2, indFrame])

    HaveNN = np.zeros((imgH, imgW, num_candidate))
    imgKeySourceFrameFlowNN = None
    if nonLocal:
        HaveKeySourceFrameFlowNN, imgKeySourceFrameFlowNN = \
            get_KeySourceFrame_flowNN(sub,
                                      indFrame,
                                      mask,
                                      videoNonLocalFlowB,
                                      videoNonLocalFlowF,
                                      video,
                                      consistencyThres)

        HaveNN[:, :, 2] = HaveKeySourceFrameFlowNN[:, :, 0] == 1
        HaveNN[:, :, 3] = HaveKeySourceFrameFlowNN[:, :, 1] == 1
        HaveNN[:, :, 4] = HaveKeySourceFrameFlowNN[:, :, 2] == 1

    HaveNN[:, :, 0] = HaveFlowNN[:, :, indFrame, 0] == 1
    HaveNN[:, :, 1] = HaveFlowNN[:, :, indFrame, 1] == 1

    NotHaveNN = np.logical_and(np.invert(HaveNN.astype(np.bool)),
            np.repeat(np.expand_dims((mask[:, :, indFrame]), 2), num_candidate, axis=2))

    if nonLocal:
        HaveNN_sum = np.logical_or.reduce((HaveNN[:, :, 0],
                                           HaveNN[:, :, 1],
                                           HaveNN[:, :, 2],
                                           HaveNN[:, :, 3],
                                           HaveNN[:, :, 4]))
    else:
        HaveNN_sum = np.logical_or.reduce((HaveNN[:, :, 0],
                                           HaveNN[:, :, 1]))
    return HaveNN, NotHaveNN, HaveNN_sum, imgKeySourceFrameFlowNN

def calculatePixel(indFrame, videoBN, videoFN, imgW, imgH, num_candidate, consistencyMap, imgKeySourceFrameFlowNN, HaveNN, NotHaveNN, HaveNN_sum, mask, video, nonLocal, alpha):
    videoCandidate = np.zeros((imgH, imgW, 3, num_candidate))
    videoCandidate[:, :, :, 0] = videoBN[:, :, :, indFrame]
    videoCandidate[:, :, :, 1] = videoFN[:, :, :, indFrame]

    if nonLocal:
        videoCandidate[:, :, :, 2] = imgKeySourceFrameFlowNN[:, :, :, 0]
        videoCandidate[:, :, :, 3] = imgKeySourceFrameFlowNN[:, :, :, 1]
        videoCandidate[:, :, :, 4] = imgKeySourceFrameFlowNN[:, :, :, 2]

    consistencyMap[:, :, :, indFrame] = np.exp( - consistencyMap[:, :, :, indFrame] / alpha)

    consistencyMap[NotHaveNN[:, :, 0], 0, indFrame] = 0
    consistencyMap[NotHaveNN[:, :, 1], 1, indFrame] = 0

    if nonLocal:
        consistencyMap[NotHaveNN[:, :, 2], 2, indFrame] = 0
        consistencyMap[NotHaveNN[:, :, 3], 3, indFrame] = 0
        consistencyMap[NotHaveNN[:, :, 4], 4, indFrame] = 0

    # weights = (consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]) / ((consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True) + 1e-16)
    weights = (consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]) / ((consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True))

    # Fix the numerical issue. 0 / 0
    fix = np.where((consistencyMap[HaveNN_sum, :, indFrame] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True) == 0)[0]
    weights[fix, :] = HaveNN[HaveNN_sum, :][fix, :] / HaveNN[HaveNN_sum, :][fix, :].sum(axis=1, keepdims=True)

    # Fuse RGB channel independently
    video[HaveNN_sum, 0, indFrame] = \
        np.sum(np.multiply(videoCandidate[HaveNN_sum, 0, :], weights), axis=1)
    video[HaveNN_sum, 1, indFrame] = \
        np.sum(np.multiply(videoCandidate[HaveNN_sum, 1, :], weights), axis=1)
    video[HaveNN_sum, 2, indFrame] = \
        np.sum(np.multiply(videoCandidate[HaveNN_sum, 2, :], weights), axis=1)
    return np.logical_and(np.invert(HaveNN_sum), mask[:, :, indFrame]), video

def map_backwards(indFrame, flowNN, videoBN, sub):
    # Index of missing pixel whose backward flow neighbor is from frame indFrame
    SourceFmInd = np.where(flowNN[:, 2, 0] == indFrame)

    print("{0:8d} pixels are from source Frame {1:3d}"
                    .format(len(SourceFmInd[0]), indFrame))
    # The location of the missing pixel whose backward flow neighbor is
    # from frame indFrame flowNN[SourceFmInd, 0, 0], flowNN[SourceFmInd, 1, 0]

    if len(SourceFmInd[0]) != 0:

        # |--------------------|
        # |       y            |
        # |   x   *            |
        # |                    |
        # |--------------------|
        # sub: numPix x 3 [y, x, t]
        # img: [y, x]
        # interp(img, x, y)

        videoBN[sub[SourceFmInd[0], :][:, 0],
                sub[SourceFmInd[0], :][:, 1],
             :, sub[SourceFmInd[0], :][:, 2]] = \
            interp(videoBN[:, :, :, indFrame],
                    flowNN[SourceFmInd, 1, 0].reshape(-1),
                    flowNN[SourceFmInd, 0, 0].reshape(-1))

        assert(((sub[SourceFmInd[0], :][:, 2] - indFrame) <= 0).sum() == 0)
    return videoBN, flowNN

def map_forward(indFrame, flowNN, videoFN, sub):
    # Index of missing pixel whose forward flow neighbor is from frame indFrame
    SourceFmInd = np.where(flowNN[:, 2, 1] == indFrame)
    print("{0:8d} pixels are from source Frame {1:3d}"
                    .format(len(SourceFmInd[0]), indFrame))
    if len(SourceFmInd[0]) != 0:

        videoFN[sub[SourceFmInd[0], :][:, 0],
                sub[SourceFmInd[0], :][:, 1],
             :, sub[SourceFmInd[0], :][:, 2]] = \
            interp(videoFN[:, :, :, indFrame],
                     flowNN[SourceFmInd, 1, 1].reshape(-1),
                     flowNN[SourceFmInd, 0, 1].reshape(-1))

        assert(((indFrame - sub[SourceFmInd[0], :][:, 2]) <= 0).sum() == 0)
    return videoFN, flowNN

def forward_pass(indFrame, NN_idx, imgW, imgH, numPixInd, videoFlowF, videoFlowB, consistencyMap, consistency_uv, mask, flowNN, HaveFlowNN, sub, args):
    # Bool indicator of missing pixels at frame t
    holepixPosInd = (sub[:, 2] == indFrame)

    # Hole pixel location at frame t, i.e. [y, x, t]
    holepixPos = sub[holepixPosInd, :]

    # Calculate the backward flow neighbor. Should be located at frame t-1
    flowB_neighbor = copy.deepcopy(holepixPos)
    flowB_neighbor = flowB_neighbor.astype(np.float32)

    flowB_vertical = videoFlowB[:, :, 1, indFrame - 1]  # t --> t-1
    flowB_horizont = videoFlowB[:, :, 0, indFrame - 1]
    flowF_vertical = videoFlowF[:, :, 1, indFrame - 1]  # t-1 --> t
    flowF_horizont = videoFlowF[:, :, 0, indFrame - 1]

    flowB_neighbor[:, 0] += flowB_vertical[holepixPos[:, 0], holepixPos[:, 1]]
    flowB_neighbor[:, 1] += flowB_horizont[holepixPos[:, 0], holepixPos[:, 1]]
    flowB_neighbor[:, 2] -= 1

    # Round the backward flow neighbor location
    flow_neighbor_int = np.round(copy.deepcopy(flowB_neighbor)).astype(np.int32)

    # Chen: I should combine the following two operations together
    # Check the backward/forward consistency
    IsConsist, _ = BFconsistCheck(flowB_neighbor,
                                  flowF_vertical,
                                  flowF_horizont,
                                  holepixPos,
                                  args.consistencyThres)

    BFdiff, BF_uv = consistCheck(videoFlowF[:, :, :, indFrame - 1],
                                 videoFlowB[:, :, :, indFrame - 1])

    # Check out-of-boundary
    # Last column and last row does not have valid gradient
    ValidPos = np.logical_and(
        np.logical_and(flow_neighbor_int[:, 0] >= 0,
                       flow_neighbor_int[:, 0] <= imgH - 1),
        np.logical_and(flow_neighbor_int[:, 1] >= 0,
                       flow_neighbor_int[:, 1] <= imgW - 1))

    # Only work with pixels that are not out-of-boundary
    holepixPos = holepixPos[ValidPos, :]
    flowB_neighbor = flowB_neighbor[ValidPos, :]
    flow_neighbor_int = flow_neighbor_int[ValidPos, :]
    IsConsist = IsConsist[ValidPos]

    # For each missing pixel in holepixPos|[y, x, t],
    # we check its backward flow neighbor flowB_neighbor|[y', x', t-1].

    # Case 1: If mask[round(y'), round(x'), t-1] == 0,
    #         the backward flow neighbor of [y, x, t] is known.
    #         [y', x', t-1] is the backward flow neighbor.

    # KnownInd: Among all backward flow neighbors, which pixel is known.
    KnownInd = mask[flow_neighbor_int[:, 0],
                    flow_neighbor_int[:, 1],
                    indFrame - 1] == 0

    KnownIsConsist = np.logical_and(KnownInd, IsConsist)

    # We save backward flow neighbor flowB_neighbor in flowNN
    flowNN[numPixInd[holepixPos[KnownIsConsist, 0],
                     holepixPos[KnownIsConsist, 1],
                     indFrame].astype(np.int32), :, NN_idx] = \
                                            flowB_neighbor[KnownIsConsist, :]
    # flowNN[np.where(holepixPosInd == 1)[0][ValidPos][KnownIsConsist], :, 0] = \
    #                                         flowB_neighbor[KnownIsConsist, :]

    # We mark [y, x, t] in HaveFlowNN as 1
    HaveFlowNN[holepixPos[KnownIsConsist, 0],
               holepixPos[KnownIsConsist, 1],
               indFrame,
               NN_idx] = 1

    # HaveFlowNN[:, :, :, 0]
    # 0: Backward flow neighbor can not be reached
    # 1: Backward flow neighbor can be reached
    # -1: Pixels that do not need to be completed

    consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 0, indFrame] = np.abs(BF_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 0])
    consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 1, indFrame] = np.abs(BF_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 1])

    # Case 2: If mask[round(y'), round(x'), t-1] == 1,
    #  the pixel@[round(y'), round(x'), t-1] is also occluded.
    #  We further check if we already assign a backward flow neighbor for the backward flow neighbor
    #  If HaveFlowNN[round(y'), round(x'), t-1] == 0,
    #   this is isolated pixel. Do nothing.
    #  If HaveFlowNN[round(y'), round(x'), t-1] == 1,
    #   we can borrow the value and refine it.

    UnknownInd = np.invert(KnownInd)

    # If we already assign a backward flow neighbor@[round(y'), round(x'), t-1]
    HaveNNInd = HaveFlowNN[flow_neighbor_int[:, 0],
                           flow_neighbor_int[:, 1],
                           indFrame - 1,
                           NN_idx] == 1

    # Unknown & IsConsist & HaveNNInd
    Valid_ = np.logical_and.reduce((UnknownInd, HaveNNInd, IsConsist))

    refineVec = np.concatenate((
        (flowB_neighbor[:, 0] - flow_neighbor_int[:, 0]).reshape(-1, 1),
        (flowB_neighbor[:, 1] - flow_neighbor_int[:, 1]).reshape(-1, 1),
        np.zeros((flowB_neighbor[:, 0].shape[0])).reshape(-1, 1)), 1)

    # Check if the transitive backward flow neighbor of [y, x, t] is known.
    # Sometimes after refinement, it is no longer known.
    flowNN_tmp = copy.deepcopy(flowNN[numPixInd[flow_neighbor_int[:, 0],
                                                flow_neighbor_int[:, 1],
                                                indFrame - 1].astype(np.int32), :, NN_idx] + refineVec[:, :])
    flowNN_tmp = np.round(flowNN_tmp).astype(np.int32)

    # Check out-of-boundary. flowNN_tmp may be out-of-boundary
    ValidPos_ = np.logical_and(
        np.logical_and(flowNN_tmp[:, 0] >= 0,
                       flowNN_tmp[:, 0] <= imgH - 1),
        np.logical_and(flowNN_tmp[:, 1] >= 0,
                       flowNN_tmp[:, 1] <= imgW - 1))

    # Change the out-of-boundary value to 0, in order to run mask[y,x,t]
    # in the next line. It won't affect anything as ValidPos_ is saved already
    flowNN_tmp[np.invert(ValidPos_), :] = 0
    ValidNN = mask[flowNN_tmp[:, 0],
                   flowNN_tmp[:, 1],
                   flowNN_tmp[:, 2]] == 0

    # Valid = np.logical_and.reduce((Valid_, ValidNN, ValidPos_))
    Valid = np.logical_and.reduce((Valid_, ValidPos_))

    # We save the transitive backward flow neighbor flowB_neighbor in flowNN
    flowNN[numPixInd[holepixPos[Valid, 0],
                     holepixPos[Valid, 1],
                     indFrame].astype(np.int32), :, NN_idx] = \
    flowNN[numPixInd[flow_neighbor_int[Valid, 0],
                     flow_neighbor_int[Valid, 1],
                     indFrame - 1].astype(np.int32), :, NN_idx] + refineVec[Valid, :]

    # We mark [y, x, t] in HaveFlowNN as 1
    HaveFlowNN[holepixPos[Valid, 0],
               holepixPos[Valid, 1],
               indFrame,
               NN_idx] = 1

    consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 0, indFrame] = np.maximum(np.abs(BF_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 0]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 0, indFrame - 1]))
    consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 1, indFrame] = np.maximum(np.abs(BF_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 1]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 1, indFrame - 1]))

    consistencyMap[:, :, NN_idx, indFrame] = (consistency_uv[:, :, NN_idx, 0, indFrame] ** 2 + consistency_uv[:, :, NN_idx, 1, indFrame] ** 2) ** 0.5

    print("Frame {0:3d}: {1:8d} + {2:8d} = {3:8d}"
    .format(indFrame,
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] == 1),
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] == 0),
                np.sum(HaveFlowNN[:, :, indFrame, NN_idx] != 99999)))
    return consistencyMap, consistency_uv, flowNN, HaveFlowNN

def backwards_pass(indFrame, NN_idx, imgW, imgH, numPixInd, videoFlowF, videoFlowB, consistencyMap, consistency_uv, mask, flowNN, HaveFlowNN, sub, args):
    # Bool indicator of missing pixels at frame t
    holepixPosInd = (sub[:, 2] == indFrame)

    # Hole pixel location at frame t, i.e. [y, x, t]
    holepixPos = sub[holepixPosInd, :]

    # Calculate the forward flow neighbor. Should be located at frame t+1
    flowF_neighbor = copy.deepcopy(holepixPos)
    flowF_neighbor = flowF_neighbor.astype(np.float32)

    flowF_vertical = videoFlowF[:, :, 1, indFrame]  # t --> t+1
    flowF_horizont = videoFlowF[:, :, 0, indFrame]
    flowB_vertical = videoFlowB[:, :, 1, indFrame]  # t+1 --> t
    flowB_horizont = videoFlowB[:, :, 0, indFrame]

    flowF_neighbor[:, 0] += flowF_vertical[holepixPos[:, 0], holepixPos[:, 1]]
    flowF_neighbor[:, 1] += flowF_horizont[holepixPos[:, 0], holepixPos[:, 1]]
    flowF_neighbor[:, 2] += 1

    # Round the forward flow neighbor location
    flow_neighbor_int = np.round(copy.deepcopy(flowF_neighbor)).astype(np.int32)

    # Check the forawrd/backward consistency
    IsConsist, _ = FBconsistCheck(flowF_neighbor,
                                  flowB_vertical,
                                  flowB_horizont,
                                  holepixPos,
                                  args.consistencyThres)

    FBdiff, FB_uv = consistCheck(videoFlowB[:, :, :, indFrame],
                                 videoFlowF[:, :, :, indFrame])

    # Check out-of-boundary
    # Last column and last row does not have valid gradient
    ValidPos = np.logical_and(
        np.logical_and(flow_neighbor_int[:, 0] >= 0,
                       flow_neighbor_int[:, 0] <= imgH - 1),
        np.logical_and(flow_neighbor_int[:, 1] >= 0,
                       flow_neighbor_int[:, 1] <= imgW - 1))

    # Only work with pixels that are not out-of-boundary
    holepixPos = holepixPos[ValidPos, :]
    flowF_neighbor = flowF_neighbor[ValidPos, :]
    flow_neighbor_int = flow_neighbor_int[ValidPos, :]
    IsConsist = IsConsist[ValidPos]

    # Case 1:
    KnownInd = mask[flow_neighbor_int[:, 0],
                    flow_neighbor_int[:, 1],
                    indFrame + 1] == 0

    KnownIsConsist = np.logical_and(KnownInd, IsConsist)
    flowNN[numPixInd[holepixPos[KnownIsConsist, 0],
                     holepixPos[KnownIsConsist, 1],
                     indFrame].astype(np.int32), :, NN_idx] = \
                                            flowF_neighbor[KnownIsConsist, :]

    HaveFlowNN[holepixPos[KnownIsConsist, 0],
               holepixPos[KnownIsConsist, 1],
               indFrame,
               NN_idx] = 1

    consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 0, indFrame] = np.abs(FB_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 0])
    consistency_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], NN_idx, 1, indFrame] = np.abs(FB_uv[holepixPos[KnownIsConsist, 0], holepixPos[KnownIsConsist, 1], 1])

    # Case 2:
    UnknownInd = np.invert(KnownInd)
    HaveNNInd = HaveFlowNN[flow_neighbor_int[:, 0],
                           flow_neighbor_int[:, 1],
                           indFrame + 1,
                           NN_idx] == 1

    # Unknown & IsConsist & HaveNNInd
    Valid_ = np.logical_and.reduce((UnknownInd, HaveNNInd, IsConsist))

    refineVec = np.concatenate((
        (flowF_neighbor[:, 0] - flow_neighbor_int[:, 0]).reshape(-1, 1),
        (flowF_neighbor[:, 1] - flow_neighbor_int[:, 1]).reshape(-1, 1),
        np.zeros((flowF_neighbor[:, 0].shape[0])).reshape(-1, 1)), 1)

    # Check if the transitive backward flow neighbor of [y, x, t] is known.
    # Sometimes after refinement, it is no longer known.
    flowNN_tmp = copy.deepcopy(flowNN[numPixInd[flow_neighbor_int[:, 0],
                                                flow_neighbor_int[:, 1],
                                                indFrame + 1].astype(np.int32), :, NN_idx] + refineVec[:, :])
    flowNN_tmp = np.round(flowNN_tmp).astype(np.int32)

    # Check out-of-boundary. flowNN_tmp may be out-of-boundary
    ValidPos_ = np.logical_and(
        np.logical_and(flowNN_tmp[:, 0] >= 0,
                       flowNN_tmp[:, 0] <= imgH - 1),
        np.logical_and(flowNN_tmp[:, 1] >= 0,
                       flowNN_tmp[:, 1] <= imgW - 1))

    # Change the out-of-boundary value to 0, in order to run mask[y,x,t]
    # in the next line. It won't affect anything as ValidPos_ is saved already
    flowNN_tmp[np.invert(ValidPos_), :] = 0
    ValidNN = mask[flowNN_tmp[:, 0],
                   flowNN_tmp[:, 1],
                   flowNN_tmp[:, 2]] == 0

    # Valid = np.logical_and.reduce((Valid_, ValidNN, ValidPos_))
    Valid = np.logical_and.reduce((Valid_, ValidPos_))

    # We save the transitive backward flow neighbor flowB_neighbor in flowNN
    flowNN[numPixInd[holepixPos[Valid, 0],
                     holepixPos[Valid, 1],
                     indFrame].astype(np.int32), :, NN_idx] = \
    flowNN[numPixInd[flow_neighbor_int[Valid, 0],
                     flow_neighbor_int[Valid, 1],
                     indFrame + 1].astype(np.int32), :, NN_idx] + refineVec[Valid, :]

    # We mark [y, x, t] in HaveFlowNN as 1
    HaveFlowNN[holepixPos[Valid, 0],
               holepixPos[Valid, 1],
               indFrame,
               NN_idx] = 1

    consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 0, indFrame] = np.maximum(np.abs(FB_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 0]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 0, indFrame + 1]))
    consistency_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], NN_idx, 1, indFrame] = np.maximum(np.abs(FB_uv[holepixPos[Valid, 0], holepixPos[Valid, 1], 1]), np.abs(consistency_uv[flow_neighbor_int[Valid, 0], flow_neighbor_int[Valid, 1], NN_idx, 1, indFrame + 1]))

    consistencyMap[:, :, NN_idx, indFrame] = (consistency_uv[:, :, NN_idx, 0, indFrame] ** 2 + consistency_uv[:, :, NN_idx, 1, indFrame] ** 2) ** 0.5

    print("Frame {0:3d}: {1:8d} + {2:8d} = {3:8d}"
    .format(indFrame,
            np.sum(HaveFlowNN[:, :, indFrame, NN_idx] == 1),
            np.sum(HaveFlowNN[:, :, indFrame, NN_idx] == 0),
            np.sum(HaveFlowNN[:, :, indFrame, NN_idx] != 99999)))
    return consistencyMap, consistency_uv, flowNN, HaveFlowNN

def get_flowNN(args,
               video,
               mask,
               videoFlowF,
               videoFlowB,
               videoNonLocalFlowF,
               videoNonLocalFlowB,
               executor):

    # video:      imgH x imgW x 3 x nFrame
    # mask:       imgH x imgW x nFrame
    # videoFlowF: imgH x imgW x 2 x (nFrame - 1)
    # videoFlowB: imgH x imgW x 2 x (nFrame - 1)
    # videoNonLocalFlowF: imgH x imgW x 2 x 3 x nFrame

    if args.Nonlocal:
        num_candidate = 5
    else:
        num_candidate = 2
    imgH, imgW, nFrame = mask.shape
    numPix = np.sum(mask)

    # |--------------------|
    # |       y            |
    # |   x   *            |
    # |                    |
    # |--------------------|

    # sub: numPix * [y x t]
    sub = np.concatenate((np.where(mask == 1)[0].reshape(-1, 1),
                          np.where(mask == 1)[1].reshape(-1, 1),
                          np.where(mask == 1)[2].reshape(-1, 1)), axis=1)

    # flowNN:      numPix x 3 x 2
    # HaveFlowNN:  imgH x imgW x nFrame x 2
    # First channel stores backward flow neighbor,
    # Second channel stores forward flow neighbor.
    # numPixInd:   imgH x imgW x nFrame

    flowNN = np.ones((numPix, 3, 2)) * 99999
    HaveFlowNN = np.ones((imgH, imgW, nFrame, 2)) * 99999
    HaveFlowNN[mask, :] = 0
    numPixInd = np.ones((imgH, imgW, nFrame)) * -1
    consistencyMap = np.zeros((imgH, imgW, num_candidate, nFrame))
    consistency_uv = np.zeros((imgH, imgW, 2, 2, nFrame))

    # numPixInd[x, y, t] gives the index of the missing pixel@[x, y, t] in sub,
    # i.e. which row. numPixInd[x, y, t] = idx; sub[idx, :] = [x, y, t]
    for idx in range(len(sub)):
        numPixInd[sub[idx, 0], sub[idx, 1], sub[idx, 2]] = idx

    # Initialization
    frameIndSetF = range(1, nFrame)
    frameIndSetB = range(nFrame - 2, -1, -1)

    # 1. Forward Pass (backward flow propagation)
    print('Forward Pass......')

    NN_idx = 0 # BN:0
    #for consistencyMap, consistency_uv, flowNN, HaveFlowNN in executor.map(forward_pass, frameIndSetF, repeat(NN_idx), repeat(imgW), repeat(imgH), repeat(numPixInd), repeat(videoFlowF), repeat(videoFlowB), repeat(consistencyMap), repeat(consistency_uv), repeat(mask), repeat(flowNN), repeat(HaveFlowNN), repeat(sub), repeat(args)):
    #    consistencyMap = consistencyMap
    for indFrame in frameIndSetF:
        consistencyMap, consistency_uv, flowNN, HaveFlowNN = forward_pass(indFrame, NN_idx, imgW, imgH, numPixInd, videoFlowF, videoFlowB, consistencyMap, consistency_uv, mask, flowNN, HaveFlowNN, sub, args);

    # 2. Backward Pass (forward flow propagation)
    print('Backward Pass......')

    NN_idx = 1 # FN:1
    #for consistencyMap, consistency_uv, flowNN, HaveFlowNN in executor.map(backwards_pass, frameIndSetB, repeat(NN_idx), repeat(imgW), repeat(imgH), repeat(numPixInd), repeat(videoFlowF), repeat(videoFlowB), repeat(consistencyMap), repeat(consistency_uv), repeat(mask), repeat(flowNN), repeat(HaveFlowNN), repeat(sub), repeat(args)):
    #    consistencyMap = consistencyMap
    for indFrame in frameIndSetB:
        consistencyMap, consistency_uv, flowNN, HaveFlowNN = backwards_pass(indFrame, NN_idx, imgW, imgH, numPixInd, videoFlowF, videoFlowB, consistencyMap, consistency_uv, mask, flowNN, HaveFlowNN, sub, args);
        

    # Interpolation
    videoBN = copy.deepcopy(video)
    videoFN = copy.deepcopy(video)
    ids = [n for n in range(nFrame)]

    for videoBN, flowNN in executor.map(map_backwards, ids, repeat(flowNN), repeat(videoBN), repeat(sub)):
        videoBN = videoBN
    #for indFrame in range(nFrame):
        #map_backwards(indFrame, flowNN, videoBN)
    
    for videoFN, flowNN in executor.map(map_forward, ids, repeat(flowNN), repeat(videoFN), repeat(sub)):
        videoFN = videoFN
    #for indFrame in range(nFrame - 1, -1, -1):
        #map_forward(indFrame, flowNN, videoFN)

    # New mask
    mask_tofill = np.zeros((imgH, imgW, nFrame)).astype(np.bool)
    
    #for res, video in executor.map(calculateMask, ids, repeat(mask_tofill), repeat(imgW), repeat(imgH), repeat(num_candidate), repeat(HaveFlowNN), repeat(consistencyMap), repeat(videoNonLocalFlowF), repeat(videoNonLocalFlowB), repeat(mask), repeat(video), repeat(videoBN), repeat(videoFN), repeat(sub), repeat(args)):
        #mask_tofill[res, indFrame] = True
    for indFrame in range(nFrame):
        res, video = calculateMask(indFrame, mask_tofill, imgW, imgH, num_candidate, HaveFlowNN, consistencyMap, videoNonLocalFlowF, videoNonLocalFlowB, mask, video, videoBN, videoFN, sub, args)
        mask_tofill[res, indFrame] = True

    return video, mask_tofill, HaveFlowNN
