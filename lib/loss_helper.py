# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

def compute_vote_loss(data_dict):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        data_dict: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = data_dict["seed_xyz"].shape[0]
    num_seed = data_dict["seed_xyz"].shape[1]  # B,num_seed,3
    vote_xyz = data_dict["vote_xyz"]  # B,num_seed*vote_factor,3
    seed_inds = data_dict["seed_inds"].long()  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(data_dict["vote_label_mask"], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(data_dict["vote_label"], 1, seed_inds_expand)
    seed_gt_votes += data_dict["seed_xyz"].repeat(1, 1, 3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size * num_seed, -1,
                                     3)  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size * num_seed, GT_VOTE_FACTOR,
                                               3)  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
    return vote_loss

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals based on center distance.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict["aggregated_vote_xyz"]
    gt_center = data_dict["center_label"][:, :, 0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B, K)).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict["objectness_scores"]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction="none")
    objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

    # Set assignment
    object_assignment = ind1  # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(data_dict, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        data_dict: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    mean_size_arr = config.mean_size_arr

    object_assignment = data_dict["object_assignment"]
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = data_dict["center"]
    gt_center = data_dict["center_label"][:, :, 0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center)  # dist1: BxK, dist2: BxK2
    box_label_mask = data_dict["box_label_mask"]
    objectness_label = data_dict["objectness_label"].float()
    centroid_reg_loss1 = \
        torch.sum(dist1 * objectness_label) / (torch.sum(objectness_label) + 1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2 * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(data_dict["heading_class_label"], 1,
                                       object_assignment)  # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction="none")
    heading_class_loss = criterion_heading_class(data_dict["heading_scores"].transpose(2, 1),
                                                 heading_class_label)  # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    heading_residual_label = torch.gather(data_dict["heading_residual_label"], 1,
                                          object_assignment)  # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                   1)  # src==1 so it"s *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(
        torch.sum(data_dict["heading_residuals_normalized"] * heading_label_one_hot,
                  -1) - heading_residual_normalized_label, delta=1.0)  # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)

    # Compute size loss
    size_class_label = torch.gather(data_dict["size_class_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction="none")
    size_class_loss = criterion_size_class(data_dict["size_scores"].transpose(2, 1), size_class_label)  # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    size_residual_label = torch.gather(data_dict["size_residual_label"], 1,
                                       object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1)  # src==1 so it"s *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(
        data_dict["size_residuals_normalized"] * size_label_one_hot_tiled, 2)  # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
        0)  # (1,1,num_size_cluster,3)
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
    size_residual_normalized_loss = torch.mean(
        huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0),
        -1)  # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(data_dict["sem_cls_label"], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction="none")
    sem_cls_loss = criterion_sem_cls(data_dict["sem_cls_scores"].transpose(2, 1), sem_cls_label)  # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def compute_cap_loss(data_dict):

    pred_caps = data_dict["lang_cap"]
    num_words = pred_caps.size(1)

    target_caps = data_dict["lang_ids"][:, 1:num_words + 1]

    _, _, num_vocabs = pred_caps.shape

    assert pred_caps.shape[0:2] == target_caps.shape[0:2], print('ERROR!!! pred {} and tgt {} shape mismatch!'.format(
        pred_caps.shape[0:2], target_caps.shape[0:2]
    ))
    # caption loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

    # mask out bad boxes
    good_bbox_masks = data_dict["good_bbox_masks"].unsqueeze(1).repeat(1, num_words)

    good_bbox_masks = good_bbox_masks.reshape(-1)
    cap_loss = torch.sum(cap_loss * good_bbox_masks) / (torch.sum(good_bbox_masks) + 1e-6)
    num_good_bbox = data_dict["good_bbox_masks"].sum()

    if num_good_bbox > 0:  # only apply loss on the good boxes
        pred_caps = pred_caps[data_dict["good_bbox_masks"]]  # num_good_bbox
        target_caps = target_caps[data_dict["good_bbox_masks"]]  # num_good_bbox

        # caption acc
        pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1)  # num_good_bbox * (num_words - 1)

        target_caps = target_caps.reshape(-1)  # num_good_bbox * (num_words - 1)
        masks = target_caps != 0
        masked_pred_caps = pred_caps[masks]
        masked_target_caps = target_caps[masks]
        cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()

    else:  # zero placeholder if there is no good box
        cap_acc = torch.zeros(1)[0].cuda()

    return cap_loss, cap_acc

def compute_relation_loss(data_dict):
    """ Compute relation loss in formula (2) of paper "Spatiality-guided Transformer for 3D Dense Captioning on Point Clouds".

    Args:
        data_dict: dict (read-only)

    Returns:
        x_loss, y_loss, z_loss, x_acc, y_acc, z_acc
    """

    object_assignment = data_dict["object_assignment"]
    _, K = object_assignment.shape
    M = data_dict["y_label"].shape[1]

    # generate gt label
    x_label = torch.gather(data_dict["x_label"], 1, object_assignment.unsqueeze(-1).repeat(1, 1, M))
    x_label = torch.gather(x_label, 2, object_assignment.unsqueeze(-2).repeat(1,K,1))
    y_label = torch.gather(data_dict["y_label"], 1, object_assignment.unsqueeze(-1).repeat(1,1,M))
    y_label = torch.gather(y_label, 2, object_assignment.unsqueeze(-2).repeat(1,K,1))
    z_label = torch.gather(data_dict["z_label"], 1, object_assignment.unsqueeze(-1).repeat(1, 1, M))
    z_label = torch.gather(z_label, 2, object_assignment.unsqueeze(-2).repeat(1,K,1))

    box_label_mask = torch.gather(data_dict['box_label_mask_int'], 1, object_assignment) & data_dict["objectness_label"]
    box_label_mask_column = torch.repeat_interleave(box_label_mask, box_label_mask.sum(1), 0)
    x_label = x_label[box_label_mask==1][box_label_mask_column==1]
    y_label = y_label[box_label_mask==1][box_label_mask_column==1]
    z_label = z_label[box_label_mask==1][box_label_mask_column==1]

    # get predicted relation
    x_pred = data_dict['relation_pred'][:,:,:,0:3]
    y_pred = data_dict['relation_pred'][:,:,:,3:6]
    z_pred = data_dict['relation_pred'][:,:,:,6:9]
    x_pred = x_pred[box_label_mask==1][box_label_mask_column==1]
    y_pred = y_pred[box_label_mask==1][box_label_mask_column==1]
    z_pred = z_pred[box_label_mask==1][box_label_mask_column==1]

    # relation loss
    x_criterion = nn.CrossEntropyLoss()
    x_loss = x_criterion(x_pred, x_label)
    y_criterion = nn.CrossEntropyLoss()
    y_loss = y_criterion(y_pred, y_label)
    z_criterion = nn.CrossEntropyLoss()
    z_loss = z_criterion(z_pred, z_label)

    # relation accuracy
    x_acc = (x_pred.argmax(-1) == x_label).sum().float()/x_label.shape[0]
    y_acc = (y_pred.argmax(-1) == y_label).sum().float()/y_label.shape[0]
    z_acc = (z_pred.argmax(-1) == z_label).sum().float()/z_label.shape[0]

    return x_loss, y_loss, z_loss, x_acc, y_acc, z_acc

def get_scene_cap_loss(data_dict, device, config,
                       detection=True, caption=True, use_relation=False):

    # 1.Vote loss: distance between obj center and nearest vote; voting module
    vote_loss = compute_vote_loss(data_dict)

    # 2.Obj loss; for each proposal, check its 2-class objectness score
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
    total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
    data_dict["objectness_label"] = objectness_label
    data_dict["objectness_mask"] = objectness_mask
    data_dict["object_assignment"] = object_assignment
    data_dict["pos_ratio"] = torch.sum(objectness_label.float().to(device)) / float(
        total_num_proposal)
    data_dict["neg_ratio"] = torch.sum(objectness_mask.float()) / float(total_num_proposal) - data_dict[
        "pos_ratio"]

    # 3.Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(
        data_dict, config)
    box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss

    # objectness
    obj_pred_val = data_dict['bbox_mask']
    obj_acc = torch.sum((obj_pred_val == data_dict["objectness_label"].long()).float() * data_dict[
        "objectness_mask"]) / (torch.sum(data_dict["objectness_mask"]) + 1e-6)
    data_dict["obj_acc"] = obj_acc

    # relation
    if use_relation:
        x_loss, y_loss, z_loss, x_acc, y_acc, z_acc = compute_relation_loss(data_dict)
        relation_loss = y_loss+z_loss+x_loss
        data_dict["x_loss"] = x_loss
        data_dict["y_loss"] = y_loss
        data_dict["z_loss"] = z_loss
        data_dict["relation_loss"] = relation_loss
        data_dict["x_acc"] = x_acc
        data_dict["y_acc"] = y_acc
        data_dict["z_acc"] = z_acc
    else:
        data_dict["x_loss"] = torch.zeros(1)[0].to(device)
        data_dict["y_loss"] = torch.zeros(1)[0].to(device)
        data_dict["z_loss"] = torch.zeros(1)[0].to(device)
        data_dict["relation_loss"] = torch.zeros(1)[0].to(device)
        data_dict["x_acc"] = torch.zeros(1)[0].to(device)
        data_dict["y_acc"] = torch.zeros(1)[0].to(device)
        data_dict["z_acc"] = torch.zeros(1)[0].to(device)

    if detection:
        data_dict["vote_loss"] = vote_loss
        data_dict["objectness_loss"] = objectness_loss
        data_dict["center_loss"] = center_loss
        data_dict["heading_cls_loss"] = heading_cls_loss
        data_dict["heading_reg_loss"] = heading_reg_loss
        data_dict["size_cls_loss"] = size_cls_loss
        data_dict["size_reg_loss"] = size_reg_loss
        data_dict["sem_cls_loss"] = sem_cls_loss
        data_dict["box_loss"] = box_loss
    else:
        data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        data_dict["center_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["size_reg_loss"] = torch.zeros(1)[0].to(device)
        data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        data_dict["box_loss"] = torch.zeros(1)[0].to(device)
        data_dict["det_loss"] = torch.zeros(1)[0].to(device)

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict)
        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].to(device)
        data_dict["cap_acc"] = torch.zeros(1)[0].to(device)
        data_dict["pred_ious"] = torch.zeros(1)[0].to(device)

    # Final loss function
    loss = 0
    if detection:
        data_dict["det_loss"] = data_dict["vote_loss"] + 0.5 * data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1 * data_dict["sem_cls_loss"]
        loss += 10*data_dict["det_loss"] # amplify
    if caption:
        loss += data_dict["cap_loss"]

    if use_relation:
        loss += 0.1*data_dict['relation_loss']

    data_dict["loss"] = loss

    return data_dict

