import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.config import CONF
from lib.ap_helper import parse_predictions, softmax
from lib.loss_helper import get_scene_cap_loss
from utils.box_util import box3d_iou_batch_tensor

# constants
DC = ScannetDatasetConfig()

def prepare_corpus(raw_data, max_len=CONF.TRAIN.MAX_DES_LEN):
    corpus = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        object_id = data["object_id"]
        object_name = data["object_name"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)

        if key not in corpus:
            corpus[key] = []  # a list of string descriptions

        corpus[key].append(description)

    return corpus

def decode_caption(raw_caption, idx2word):
    decoded = ["sos"]
    for token_idx in raw_caption:
        token_idx = token_idx.item()
        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded

def check_candidates(corpus, candidates):
    placeholder = "sos eos"
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates

def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates

def feed_scene_cap(model, device, dataset, dataloader, folder, is_eval=True,
                   min_iou=CONF.EVAL.MIN_IOU_THRESHOLD, organized=None, eval_tag=None,
                   save_encoder_attn=False, save_decoder_attn=False,
                   save_proposal=False):

    candidates = {}
    intermediates = {}
    interm_proposal_related = {}

    interm_decoder_attn_wegiths = None
    interm_encoder_attn_weights = None

    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].to(device)

        with torch.no_grad():
            data_dict = model(data_dict, is_eval)
            data_dict = get_scene_cap_loss(data_dict, device, DC, detection=True, caption=False)

        if save_encoder_attn or save_decoder_attn:
            B, K, _ = data_dict["lang_cap"].shape
            interm_decoder_attn_wegiths = []
            interm_encoder_attn_weights = []

            if save_decoder_attn:
                for layer in model.caption.model.decoder.layers:
                    bk, n_heads, cap_len, cap_len_ = layer.self_attn.attn.shape
                    assert bk == B * K
                    assert cap_len == cap_len_
                    if save_decoder_attn:
                        interm_decoder_attn_wegiths.append(layer.self_attn.attn.view(B, K, n_heads, cap_len, cap_len_).unsqueeze(0))

            if save_encoder_attn:
                for layer in model.caption.model.encoder.layers:
                    _, en_heads, nproposal, nproposal = layer.self_attn.attn.shape
                    interm_encoder_attn_weights.append(
                        layer.self_attn.attn.view(B, en_heads, nproposal, nproposal).unsqueeze(0))

            if save_decoder_attn:
                interm_decoder_attn_wegiths = torch.cat(interm_decoder_attn_wegiths, dim=0)
            if save_encoder_attn:
                interm_encoder_attn_weights = torch.cat(interm_encoder_attn_weights, dim=0)

        # unpack
        if len(data_dict["lang_cap"].shape) == 4:
            captions = data_dict["lang_cap"].argmax(-1)  # batch_size, num_proposals, max_len - 1
        else:
            assert len(data_dict["lang_cap"].shape) == 3
            captions = data_dict["lang_cap"]  # batch_size, num_proposals, max_len - 1

        dataset_ids = data_dict["dataset_idx"]  # B,1 the id from dataset.get_item(id)
        batch_size, num_proposals, _ = captions.shape

        # post-process
        # config
        POST_DICT = {
            "remove_empty_box": True,  # nonempty_box_mask
            "use_3d_nms": True,
            "nms_iou": 0.25,
            "use_old_type_nms": False,
            "cls_nms": True,
            "per_class_proposal": True,
            "conf_thresh": 0.05,
            "dataset_config": DC
        }

        # nms mask: remove empty box and perform class nms
        _ = parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.LongTensor(data_dict["pred_mask"]).cuda()  ## B, #proposals

        # objectness mask
        obj_masks = data_dict['bbox_mask'].long()

        # final mask
        nms_masks = nms_masks * obj_masks  # b,k

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1,
                                           data_dict["object_assignment"])  ##B, #proposal

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            data_dict["gt_box_corner_label"],
            1,
            data_dict["object_assignment"].view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
            ##B, #proposal, 8, 3
        )  # batch_size, num_proposals, 8, 3
        detected_bbox_corners = data_dict["bbox_corner"]  # batch_size, num_proposals, 8, 3

        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3),  # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3)  # batch_size * num_proposals, 8, 3
        ).view(batch_size, num_proposals)

        # find good boxes (IoU > threshold)
        good_bbox_masks = ious > min_iou  # batch_size, num_proposals 0.5 #b,k

        # dump generated captions
        if save_proposal:
            obj_objectness_score = softmax(data_dict['objectness_scores'].detach().cpu().numpy())[:, :, 1]
        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            valid_obj_flag = False
            for prop_id in range(num_proposals):
                if nms_masks[batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
                    valid_obj_flag = True
                    object_id = str(detected_object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], dataset.vocabulary[
                        "idx2word"])  # project from 31-length id to sos ... eos

                    try:
                        ann_list = list(organized[scene_id][object_id].keys())
                        object_name = organized[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        key = "{}|{}|{}".format(scene_id, object_id, object_name)
                        candidates[key] = [caption_decoded]  # "scene0700_00|33|whiteboard": ["sos ... eos"]

                        if save_decoder_attn or save_encoder_attn:
                            single_decoder_attn_weight = None
                            single_encoder_attn_weight = None

                            if save_decoder_attn:
                                single_decoder_attn_weight = interm_decoder_attn_wegiths[:, batch_id, prop_id, :, :, :]
                                single_decoder_attn_weight = [single_decoder_attn_weight[layer_i].unsqueeze(0).cpu() for layer_i in
                                                      range(single_decoder_attn_weight.shape[0])]

                            if save_encoder_attn:
                                single_encoder_attn_weight = interm_encoder_attn_weights[:, batch_id, :, :, :]
                                single_encoder_attn_weight = [single_encoder_attn_weight[layer_i].unsqueeze(0) for
                                                              layer_i in
                                                              range(single_encoder_attn_weight.shape[0])]

                            intermediates[key] = {'token': caption_decoded.split(" "),
                                                  'decoder_attn_weights': single_decoder_attn_weight,
                                                  'encoder_attn_weights': single_encoder_attn_weight,
                                                  'prop_id': prop_id}


                    except KeyError:
                        continue

            if valid_obj_flag and save_proposal:
                interm_proposal_related[scene_id] = {'obj_id': detected_object_ids[batch_id].cpu().numpy(),
                                                     'obj_mask': obj_masks[batch_id].cpu().numpy(),
                                                     'ious': ious[batch_id].cpu().numpy(),
                                                     'nms_mask': nms_masks[batch_id].cpu().numpy(),
                                                     'box_corners': detected_bbox_corners[batch_id].cpu().numpy(),
                                                     'class': data_dict['bbox_sems'][batch_id].cpu().numpy(),
                                                     'objectness': obj_objectness_score[batch_id],
                                                     'center': data_dict['center'][batch_id].cpu().numpy()}

    # detected boxes
    if save_decoder_attn or save_encoder_attn:
        print("saving attention weights...")
        interm_path = os.path.join(CONF.PATH.OUTPUT, folder, "attn_weights_{}.pt".format(eval_tag))
        torch.save(intermediates, interm_path)

    if save_proposal:
        print("saving proposal related info...")
        interm_path = os.path.join(CONF.PATH.OUTPUT, folder, "proposal_related_{}.pt".format(eval_tag))
        torch.save(interm_proposal_related, interm_path)

    return candidates

def eval_cap(model, device, dataset, dataloader, phase, folder,
             is_eval=True, max_len=CONF.TRAIN.MAX_DES_LEN,
             min_iou=CONF.EVAL.MIN_IOU_THRESHOLD, eval_tag=None,
             save_encoder_attn=False, save_decoder_attn=False,
             save_proposal=False):
    '''
    return bleu, cider, rouge, meteor comparing between corpus and pred for each object
    :param model: self.model
    :param device: self.device,
    :param dataset: self.dataset["eval"][phase],
    :param dataloader: self.dataloader["eval"][phase]
    :param phase: phase
    :param folder: self.stamp
    :param use_tf: False
    :param is_eval:
    :param max_len: CONF.TRAIN.MAX_DES_LEN; 30
    :param min_iou: CONF.EVAL.MIN_IOU_THRESHOLD; 0.5
    :return:
    '''
    # corpus
    corpus_path = os.path.join(CONF.PATH.OUTPUT, folder, "corpus_{}.json".format(phase))
    if not os.path.exists(corpus_path):
        print("preparing corpus...")
        if dataset.name == "ScanRefer":
            raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.json".format(phase))))
        elif dataset.name == "ReferIt3D":
            raw_data = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_{}.json".format(phase))))
        else:
            raise ValueError("Invalid dataset.")

        corpus = prepare_corpus(raw_data, max_len)  # scene_id|obj_id|obj_name: [list of string descriptions with sos and eos appended]
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)
    else:
        print("loading corpus...")
        with open(corpus_path) as f:
            corpus = json.load(f)

    if dataset.name == "ScanRefer":
        organized = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json")))
    elif dataset.name == "ReferIt3D":
        organized = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_organized.json")))
    else:
        raise ValueError("Invalid dataset.")

    if eval_tag is not None:
        pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}_{}.json".format(phase, eval_tag))
    else:
        pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}.json".format(phase))
    # generate results
    print("generating descriptions...")
    candidates = feed_scene_cap(model, device, dataset, dataloader, folder, is_eval,
                                min_iou, organized=organized, eval_tag=eval_tag,
                                save_encoder_attn=save_encoder_attn, save_decoder_attn=save_decoder_attn,
                                save_proposal=save_proposal)


    # check candidates
    # NOTE: make up the captions for the undetected object by "sos eos"
    candidates = check_candidates(corpus, candidates)
    candidates = organize_candidates(corpus, candidates)

    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # compute scores
    print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    return bleu, cider, rouge, meteor

