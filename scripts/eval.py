from datetime import datetime

# HACK ignore warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from models.SpaCapNet import SpaCapNet
from lib.dataset import ScannetReferenceDataset
from lib.config import CONF
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_scene_cap_loss
from lib.eval_helper import eval_cap, decode_caption
from lib.visualize_helper import write_bbox

# constants
DC = ScannetDatasetConfig()

def log(*args):
    if pout:
        print(' '.join(str(arg) for arg in args))

def get_dataloader(args, scanrefer):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        split="val",
        name=args.dataset,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader


def get_model(args, dataset, device, root=CONF.PATH.OUTPUT):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
        not args.no_height)
    log('use SpaCapNet; model input dimension', input_channels,
          '(multiview: {} normal: {} color: {} height: {})'.format(args.use_multiview, args.use_normal, args.use_color,
                                                                   not args.no_height))

    model = SpaCapNet(
        num_class=DC.num_class,  # 18
        vocabulary=dataset.vocabulary,  # word2idx; idx2word
        num_heading_bin=DC.num_heading_bin,  # 1
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,  # 18
        input_feature_dim=input_channels,  # #dimensions except xyz
        num_proposal=args.num_proposals,  # 256
        no_caption=args.eval_detection,
        ### transformer-related parameters
        N=args.N,
        h=args.h,
        d_model=args.d_model,
        d_ff=args.d_ff,  # 2048
        transformer_dropout=args.transformer_dropout,
        src_pos_type=args.src_pos_type if not args.no_learnt_src_pos else None,
        use_transformer_encoder=not args.no_enc,
        early_guide=not args.late_guide
    )

    # load
    model_name = "model_last.pth" if args.use_last else "model.pth"
    model_path = os.path.join(root, args.folder, model_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # to device
    model.to(device)

    # set mode
    model.eval()

    return model

def get_scannet_scene_list(data):
    scene_list = sorted(list(set([d["scene_id"] for d in data])))
    return scene_list

def get_eval_data(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "ReferIt3D":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    eval_scene_list = get_scannet_scene_list(scanrefer_train) if args.use_train else get_scannet_scene_list(
        scanrefer_val)
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(scanrefer_train[0]) if args.use_train else deepcopy(scanrefer_val[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    log("eval on {} scenes".format(len(scanrefer_eval)))

    return scanrefer_eval


def eval_caption(args):
    print('\nStart evaluating captioning at seed {}...'.format(args.seed))
    log("initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get eval data
    log("preparing data...")
    scanrefer_eval = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval)

    # get model
    model = get_model(args, dataset, device)

    # evaluate
    bleu, cider, rouge, meteor = eval_cap(model, device, dataset, dataloader, "val", args.folder,
                                          min_iou=args.min_iou, eval_tag=args.eval_tag,
                                          save_encoder_attn=args.save_encoder_attn, save_decoder_attn=args.save_decoder_attn,
                                          save_proposal=args.save_proposal,
                                          )

    # report
    report_log = 'Seed: {}\n'.format(args.seed)
    report_log += datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '\n'
    log("\n----------------------Evaluation-----------------------")
    log("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    log("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    log("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    log("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    log("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    log("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    log("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1])))
    log()

    report_log += "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][0], max(bleu[1][0]),
                                                                             min(bleu[1][0]))
    report_log += "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][1], max(bleu[1][1]),
                                                                             min(bleu[1][1]))
    report_log += "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][2], max(bleu[1][2]),
                                                                             min(bleu[1][2]))
    report_log += "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][3], max(bleu[1][3]),
                                                                             min(bleu[1][3]))
    report_log += "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(cider[0], max(cider[1]), min(cider[1]))
    report_log += "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(rouge[0], max(rouge[1]), min(rouge[1]))
    report_log += "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(meteor[0], max(meteor[1]), min(meteor[1]))
    report_log += "-"*50+'\n'
    report_log += datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '\n'

    if not args.mul_eval:
        with open(os.path.join(CONF.PATH.OUTPUT, args.folder, 'eval_caption_{}.txt'.format(args.eval_tag)), 'a') as f:
            f.write(report_log)
    else:
        return bleu[0][3], cider[0], rouge[0], meteor[0]


def eval_detection(args):
    print("Start evaluating detection at seed {}...".format(args.seed))
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    log("preparing data...")
    # get eval data
    scanrefer_eval = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval)

    # model
    log("initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, dataset, device)

    # config
    POST_DICT = {
        "remove_empty_box": True,
        "use_3d_nms": True,
        "nms_iou": 0.25,
        "use_old_type_nms": False,
        "cls_nms": True,
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    for data in tqdm(dataloader):

        for key in data:
            data[key] = data[key].to(device)

        # feed
        with torch.no_grad():
            data = model(data, True)
            data = get_scene_cap_loss(data, device, DC, detection=True, caption=False)

        batch_pred_map_cls = parse_predictions(data, POST_DICT)
        batch_gt_map_cls = parse_groundtruths(data, POST_DICT, save_time=False)
        for ap_calculator in AP_CALCULATOR_LIST:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate object detection results and report
    report_log = 'Seed: {}\n'.format(args.seed)
    report_log += datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '\n'
    ret_val = 0
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        log()
        log("-" * 10, "iou_thresh: %f" % (AP_IOU_THRESHOLDS[i]), "-" * 10)
        report_log += '{} {} {}\n'.format("-" * 10, "iou_thresh: %f" % (AP_IOU_THRESHOLDS[i]), "-" * 10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log("eval %s: %f" % (key, metrics_dict[key]))
            report_log += "eval %s: %f\n" % (key, metrics_dict[key])
            if i == 1 and key == 'mAP':
                ret_val = float(metrics_dict[key])
    report_log += "-" * 50 + '\n'
    report_log += datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '\n'

    if not args.mul_eval:
        with open(os.path.join(CONF.PATH.OUTPUT, args.folder, 'eval_detection_{}.txt'.format(args.eval_tag)), 'a') as f:
            f.write(report_log)
    else:
        return ret_val


def eval_visualize(args):
    from utils.box_util import box3d_iou_batch_tensor
    from shutil import copyfile
    from utils.colors import COLORS
    SCANNET_MESH = os.path.join(CONF.PATH.SCANNET_SCANS, "{}", "{}_axis_aligned.ply")  # scene_id, scene_id
    SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json")))

    print("Start evaluating detection at seed {} with bbox prediction stored at {}".format(args.seed, os.path.join(CONF.PATH.OUTPUT, args.folder, "vis/")))
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    # get eval data
    scanrefer_eval = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_dataloader(args, scanrefer_eval)

    # model
    print("initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, dataset, device)

    # config
    POST_DICT = {
        "remove_empty_box": True,
        "use_3d_nms": True,
        "nms_iou": 0.25,
        "use_old_type_nms": False,
        "cls_nms": True,
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    }

    for data_dict in tqdm(dataloader):
        for key in data_dict:
            data_dict[key] = data_dict[key].to(device)

        # feed
        with torch.no_grad():
            data_dict = model(data_dict, True)
            data_dict = get_scene_cap_loss(data_dict, device, DC, detection=True, caption=False)
        # unpack
        if len(data_dict["lang_cap"].shape) == 4:
            captions = data_dict["lang_cap"].argmax(-1)  # batch_size, num_proposals, max_len - 1
        else:
            assert len(data_dict["lang_cap"].shape) == 3
            captions = data_dict["lang_cap"]  # .argmax(-1) # batch_size, num_proposals, max_len - 1

        dataset_ids = data_dict["dataset_idx"]  # B,1 the id from dataset.get_item(id)
        batch_size, num_proposals, _ = captions.shape

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
        assigned_target_bbox_corners = torch.gather(data_dict["gt_box_corner_label"], 1,
                                                    data_dict["object_assignment"].view(batch_size, num_proposals, 1,
                                                                                        1).repeat(1, 1, 8, 3)
                                                    ##B, #proposal, 8, 3
                                                    )  # batch_size, num_proposals, 8, 3
        detected_bbox_corners = data_dict["bbox_corner"]  # batch_size, num_proposals, 8, 3

        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3),  # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3)  # batch_size * num_proposals, 8, 3
        ).view(batch_size, num_proposals)

        # find good boxes (IoU > threshold)
        good_bbox_masks = ious > 0.5  # batch_size, num_proposals 0.5 #b,k

        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            scene_root = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis/{}".format(scene_id))
            if args.verbose:
                print('>> scene root:', scene_root)
            if args.nodryrun:
                os.makedirs(scene_root, exist_ok=True)
            mesh_path = os.path.join(scene_root, "{}.ply".format(scene_id))
            if args.nodryrun:
                copyfile(SCANNET_MESH.format(scene_id, scene_id), mesh_path)
            candidates = {}
            for prop_id in range(num_proposals):
                if nms_masks[batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
                    object_id = str(detected_object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], dataset.vocabulary[
                        "idx2word"])  # project from 31-length id to sos ... eos
                    detected_bbox_corner = detected_bbox_corners[batch_id, prop_id].detach().cpu().numpy()
                    try:
                        ann_list = list(SCANREFER_ORGANIZED[scene_id][object_id].keys())
                        object_name = SCANREFER_ORGANIZED[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        candidates[object_id] = {
                            "object_name": object_name,
                            "description": caption_decoded
                        }

                        ply_name = "pred-{}-{}.ply".format(object_id, object_name)
                        ply_path = os.path.join(scene_root, ply_name)
                        if args.verbose:
                            print(ply_path)
                        palette_idx = int(object_id) % len(COLORS)
                        color = COLORS[palette_idx]
                        if args.nodryrun:
                            write_bbox(detected_bbox_corner, color, ply_path)
                    except KeyError:
                        continue

            # save predictions for the scene
            pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder,
                                     "vis/{}/predictions.json".format(scene_id))
            if args.verbose:
                print('pred_path:', pred_path)
            if args.nodryrun:
                with open(pred_path, "w") as f:
                    json.dump(candidates, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_tag", type=str,
                        help="tag for the evaluation, e.g. muleval")
    parser.add_argument("--mul_eval", action="store_true",
                        help="try 100 evaluations with different seedings: run eval_caption and eval_detection 100 times")
    parser.add_argument("--eval_caption", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--eval_detection", action="store_true", help="evaluate the object detection results")
    parser.add_argument("--eval_visualize", action="store_true",
                        help="save predicted bbox ply to CONF.PATH.OUTPUT/args.folder/vis/scene0011_00/...")

    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--min_iou", type=float, default=0.5, help="Min IoU threshold for evaluation")

    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_last", action="store_true", help="Use the last model")

    # Transformer
    parser.add_argument("--no_enc", action="store_true",
                        help="DO NOT use transformer encoder to aggregate visual tokens")
    parser.add_argument('--N', default=6, type=int, help='number of encoder/decoder layers')
    parser.add_argument('--h', default=8, type=int, help='multi-head number')
    parser.add_argument('--d_model', default=128, type=int, help='')
    parser.add_argument('--d_ff', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument("--no_learnt_src_pos", action="store_true", help="DO NOT use learnable src position.")
    parser.add_argument("--src_pos_type", type=str, default="xyz",
                        help="src pos embedding based on , [choices: xyz, center, loc...]")
    parser.add_argument("--late_guide", action="store_true", help="DO NOT use target obj vision token early")

    # Transformer attention visualization related
    parser.add_argument("--save_encoder_attn", action="store_true",
                        help="save transformer encoder attention to CONF.PATH.OUTPUT/args.folder/attn_weights_[args.eval_tag].pt")
    parser.add_argument("--save_decoder_attn", action="store_true",
                        help="save transformer decoder attention to CONF.PATH.OUTPUT/args.folder/attn_weights_[args.eval_tag].pt")
    parser.add_argument("--save_proposal", action="store_true",
                        help="save proposal-related info to CONF.PATH.OUTPUT/args.folder/proposal_related_[args.eval_tag].pt")

    parser.add_argument("--verbose", action="store_true",
                        help="eval_visualize: print path info")
    parser.add_argument("--nodryrun", action="store_true",
                        help="eval_visualize: actually generate file and folders")

    args = parser.parse_args()

    # setting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # evaluate
    if args.mul_eval:
        print('** MULTIPLE EVALUATION **')
        import csv

        pout=False
        metrics = np.zeros((100,5)) #c, b4, m, r, map
        csv_filed = ['seed', 'cider', 'bleu4', 'meteor', 'rouge', 'mAP']
        csv_rows = []
        for sd in tqdm(range(0, 100)):
            args.seed = sd
            # reproducibility
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(args.seed)

            bleu, cider, rouge, meteor = eval_caption(args)
            map = eval_detection(args)

            metrics[sd][0], metrics[sd][1], metrics[sd][2], metrics[sd][3], metrics[sd][4] = cider, bleu, meteor, rouge, map
            row_content = [[sd, cider, bleu, meteor, rouge, map]]
            csv_rows.extend(row_content)
            print('\nseed: {} ==> [C, B4, M, R, mAP]: {}'.format(sd, metrics[sd]))
            print('Best CIDEr at seed: {} ==> [C, B4, M, R, mAP]: {}'.format(np.argmax(metrics[:, 0]),
                                                                             metrics[np.argmax(metrics[:, 0])]))

        print('** FINISH 100 evaluations **')
        print('Best CIDEr at seed: {} ==> [C, B4, M, R, mAP]: {}'.format(np.argmax(metrics[:,0]), metrics[np.argmax(metrics[:,0])]))
        print('Check detailed 100 evaluation results at', os.path.join(CONF.PATH.OUTPUT, args.folder, 'mul_eval_results.csv'))
        with open(os.path.join(CONF.PATH.OUTPUT, args.folder, '{}_results.csv'.format(args.eval_tag)), "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_filed)
            writer.writerows(csv_rows)
    else:
        pout=True
        if args.eval_caption: eval_caption(args)
        if args.eval_detection: eval_detection(args)
        if args.eval_visualize: eval_visualize(args)