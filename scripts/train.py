# HACK ignore warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import argparse
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.SpaCapNet import SpaCapNet
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
# constants
DC = ScannetDatasetConfig()


def get_scanrefer(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "ReferIt3D":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")


    if args.no_caption:
        train_scene_list = get_scannet_scene_list("train")
        val_scene_list = get_scannet_scene_list("val")

        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)

        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_eval_val])))
        num_scenes = args.num_scenes
        if num_scenes == -1:
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes
        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]
        val_scene_list = val_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        # eval on train
        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)

        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)



    return new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val


def get_dataloader(args, scanrefer, split, augment):
    '''
    :param args:
    :param scanrefer: scanrefer_train; list of train caption annotations
    :param split: "train"
    :param augment: True
    :return:
    '''
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        split=split,
        name=args.dataset,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=augment,
        use_relation=not args.no_relation
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                            drop_last=False)

    return dataset, dataloader


def get_model(args, dataset, device):
    '''
    :param args:
    :param dataset: dataset["train"]; train dataset
    :param device: gpu0 or cpu
    :return:
    '''
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    print('model input dimension', input_channels,
          '(multiview: {} normal: {} color: {} height: {})'.format(args.use_multiview, args.use_normal, args.use_color, not args.no_height))
    model = SpaCapNet(
        num_class=DC.num_class,  # 18
        vocabulary=dataset.vocabulary,  # word2idx; idx2word
        num_heading_bin=DC.num_heading_bin,  # 1
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,  # 18
        input_feature_dim=input_channels,  # #dimensions except xyz
        num_proposal=args.num_proposals,  # 256
        no_caption=args.no_caption,
        ### transformer-related parameters
        N=args.N,
        h=args.h,
        d_model=args.d_model,
        d_ff=args.d_ff,  # 2048
        transformer_dropout=args.transformer_dropout,  # 0.1
        src_pos_type=args.src_pos_type if not args.no_learnt_src_pos else None,
        use_transformer_encoder=not args.no_enc,
        early_guide=not args.late_guide,
        check_relation=not args.no_relation,
    )

    # load pretrained model
    pretrained_model = SpaCapNet(
        num_class=DC.num_class,
        vocabulary=dataset.vocabulary,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        num_proposal=args.num_proposals,
        input_feature_dim=input_channels,
        no_caption=True,
        src_pos_type=args.src_pos_type if not args.no_learnt_src_pos else None,
    )

    pretrained_name = "PRETRAIN_VOTENET_XYZ"
    if args.use_color: pretrained_name += "_COLOR"
    if args.use_multiview: pretrained_name += "_MULTIVIEW"
    if args.use_normal: pretrained_name += "_NORMAL"
    print("loading pretrained VoteNet:", pretrained_name)
    pretrained_path = os.path.join(CONF.PATH.PRETRAINED, pretrained_name, "model.pth")
    pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

    # mount
    model.backbone_net = pretrained_model.backbone_net
    model.vgen = pretrained_model.vgen
    model.proposal = pretrained_model.proposal

    if args.no_detection:
        # freeze pointnet++ backbone
        # you can directly set model.backbone_net.requires_grad_(False)
        for param in model.backbone_net.parameters():
            param.requires_grad = False

        # freeze voting
        for param in model.vgen.parameters():
            param.requires_grad = False

        # freeze detector
        for param in model.proposal.parameters():
            param.requires_grad = False

    # multi-GPU
    print("using {} GPUs...".format(torch.cuda.device_count()))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # to device
    model.to(device)

    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataset, dataloader):
    '''
    :param args:
    :param dataset: {'train':..., 'eval': {'train':..., 'val':...}}
    :param dataloader: {'train':..., 'eval': {'train':..., 'val':...}}
    :return:
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args, dataset["train"], device)

    if args.optimizer == 'adam':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "caption" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "caption" in n and p.requires_grad],
                "lr": args.transformer_lr,
            },
        ]
        optimizer = optim.Adam(param_dicts,
                                lr=args.lr,
                                weight_decay=args.wd)
    else:
        raise NotImplementedError

    checkpoint_best = None

    start_epoch = 0
    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "model_last.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint_best = checkpoint["best"]
        start_epoch = checkpoint['epoch']+1
        print('Continue from epoch', start_epoch+1)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_" + args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.no_caption else None
    LR_DECAY_RATE = 0.1 if args.no_caption else None
    BN_DECAY_STEP = 20 if args.no_caption else None
    BN_DECAY_RATE = 0.5 if args.no_caption else None

    solver = Solver(
        model=model,
        device=device,
        config=DC,
        dataset=dataset,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,  # iterations of validating; 2000
        detection=not args.no_detection,
        caption=not args.no_caption,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        criterion=args.criterion,  # for selecting the best model
        checkpoint_best=checkpoint_best,
        no_eval_on_train=not args.eval_on_train,
        train_start_epoch=start_epoch,
        use_relation=not args.no_relation,
    )
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(dataset["train"])
    info["num_eval_train"] = len(dataset["eval"]["train"]) if args.eval_on_train else 0
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list) if args.eval_on_train else 0
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split):
    print('scannet dataset is', CONF.PATH.SCANNET_META)
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def train(args):
    # init training dataset
    print(">>> Train SpaCap <<<\n{}".format('-'*100))
    print("1. preparing data from", args.dataset)

    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val = get_scanrefer(args)

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer_train, "train", True,)
    if args.eval_on_train:
        eval_train_dataset, eval_train_dataloader = get_dataloader(args, scanrefer_eval_train, "val", False)
    eval_val_dataset, eval_val_dataloader = get_dataloader(args, scanrefer_eval_val, "val", False)

    dataset = {
        "train": train_dataset,
        "eval": {
            "train": eval_train_dataset if args.eval_on_train else None,
            "val": eval_val_dataset
        }
    }
    dataloader = {
        "train": train_dataloader,
        "eval": {
            "train": eval_train_dataloader if args.eval_on_train else None,
            "val": eval_val_dataloader
        }
    }

    print("2. initializing model, optimizer, solver")
    solver, num_params, root = get_solver(args, dataset, dataloader)
    print('num_params', num_params)
    print("3. start training\n{}\n".format('-'*100))
    save_info(args, root, num_params, dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=1000)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=2000)
    # optimization: ADAM
    parser.add_argument("--optimizer", type=str, default='adam', help="use adam as optimizer by default")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)

    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")

    parser.add_argument("--criterion", type=str, default="cider", \
                        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum]")

    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_caption", action="store_true", help="Do NOT train the caption module.")

    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use normal in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")

    # Transformer
    parser.add_argument("--no_enc", action="store_true", help="DO NOT use transformer encoder to aggregate visual tokens")
    parser.add_argument("--late_guide", action="store_true", help="DO NOT use target obj vision token early")
    parser.add_argument('--N', default=6, type=int, help='number of encoder/decoder layers')
    parser.add_argument('--h', default=8, type=int, help='multi-head number')
    parser.add_argument('--d_model', default=128, type=int, help='')
    parser.add_argument('--d_ff', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument("--no_learnt_src_pos", action="store_true", help="DO NOT use learnable src position.")
    parser.add_argument("--src_pos_type", type=str, default="xyz",
                        help="src pos embedding based on , [choices: xyz, center, ..]")
    parser.add_argument("--no_relation", action="store_true", help="DO NOT use 9-d relation guidance for transformer encoder")
    parser.add_argument("--transformer_lr", type=float,  help="learning rate for transformer part", default=1e-3)
    parser.add_argument("--eval_on_train", action="store_true", help="DO NOT save time: also eval on train")

    args = parser.parse_args()


    # setting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)