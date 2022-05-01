import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
PROCESSED_DATA_FOLDER = './scannet_data'
DATASET_FOLDER = './scans'

def get_scene_list(dataset):
    ## Generate the list of scene id in train and val splits for a dataset 'scanrefer' or 'nr3d'
    ## Generated files: '[dataset]_train_scenelist.txt' and '[dataset]_val_scenelist.txt' under './meta_data'
    print('getting scene list...')
    if os.path.isfile('./meta_data/{}_train_scenelist.txt'.format(dataset)) and os.path.isfile('./meta_data/{}_val_scenelist.txt'.format(dataset)):
        print('train/val scene lists have already been generated for', dataset)
        return
    train_json = json.load(open('../{}_train.json'.format('ScanRefer_filtered' if dataset=='scanrefer' else 'nr3d')))
    val_json = json.load(open('../{}_val.json'.format('ScanRefer_filtered' if dataset=='scanrefer' else 'nr3d')))
    train_scene_list = sorted(list(set([data["scene_id"] for data in train_json])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in val_json])))
    print('#scenes in dataset {} train split: {} val split: {}'.format(dataset, len(train_scene_list), len(val_scene_list)))

    meta_p = './meta_data'
    with open(os.path.join(meta_p, '{}_train_scenelist.txt'.format(dataset)), 'w') as f:
        f.writelines('\n'.join(train_scene_list))
    with open(os.path.join(meta_p, '{}_val_scenelist.txt'.format(dataset)), 'w') as f:
        f.writelines('\n'.join(val_scene_list))

def get_obj_id_json_per_scene(scene_id, dryrun=True):
    ## Generate json file to record object info per scene {objectId: obj_label}
    ## Generated files: '[scene_id]_obj.json' under './scans/[scene_id]'
    print('getting obj info [objectId: obj_label] for scene {}... [invoked for relation label visualization ONLY]'.format(scene_id))
    ret_dict = {}
    scene_folder = os.path.join(DATASET_FOLDER, scene_id)
    obj_file = os.path.join(scene_folder, '{}_obj.json'.format(scene_id))
    if os.path.isfile(obj_file):
        return
    agg_file = os.path.join(scene_folder, '{}.aggregation.json'.format(scene_id))
    with open(agg_file) as f:
        agg_f = json.load(f)['segGroups']
    for obj in agg_f:
        ret_dict[obj['objectId']] = obj['label']
    ret_dict = dict(sorted(ret_dict.items()))
    if not dryrun:
        with open(obj_file, "w") as f:
            json.dump(ret_dict, f, indent=4)


def get_z_relation_per_scene(scene_id, visualize, savefig, dryrun=True, verbose=True, save_npy=False):
    if os.path.isfile(os.path.join(PROCESSED_DATA_FOLDER, '{}_z.npy'.format(scene_id))) and save_npy:
        print('scene_id {} z relation gt already existed. skipped!'.format(scene_id))
        return

    upper_thresh = 0.3

    bboxes = np.load(os.path.join(PROCESSED_DATA_FOLDER, '{}_aligned_bbox.npy'.format(scene_id)))
    if not save_npy:
        bboxes = bboxes[bboxes[:, -1].argsort()]
    bboxes_zmin = bboxes[:, 2] - bboxes[:, 5]*0.5
    to_minus = np.tile(bboxes_zmin,(bboxes.shape[0],1))
    dif = bboxes_zmin[:, np.newaxis] - to_minus
    up_mask = (dif >= upper_thresh*bboxes[:,5]).astype(int)
    up_mask[(np.where(up_mask==1)[1], np.where(up_mask==1)[0])] = -1

    if save_npy:
        out_npy = np.zeros(shape=up_mask.shape, dtype=np.uint32)
        out_npy[up_mask==0] = 1
        out_npy[up_mask==-1] = 2
        out_npy[up_mask==1] = 0

        if verbose:
            print(os.path.join(PROCESSED_DATA_FOLDER, '{}_z.npy'.format(scene_id)))
        if not dryrun:
            np.save(os.path.join(PROCESSED_DATA_FOLDER, '{}_z.npy'.format(scene_id)), out_npy)
        return

    json_f = json.load(open(os.path.join(DATASET_FOLDER, scene_id, '{}_obj.json'.format(scene_id))))
    tick_label = ['{}-{}'.format(bid, json_f[str(bid)]) for bid in bboxes[:,-1].astype(int)]
    df = pd.DataFrame(up_mask, columns=tick_label, index=tick_label)
    fig = plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    res = sns.heatmap(df, annot=True, vmin=dif.min(), vmax=dif.max(),
                      fmt='.2f', cmap=cmap, cbar_kws={"shrink": .82},
                      linewidths=0.1, linecolor='gray')

    if verbose:
        print("Relation along Z-axis for {}".format(scene_id))
    plt.title("Relation along Z-axis for {}".format(scene_id))

    if savefig:
        if not dryrun:
            plt.savefig(os.path.join(DATASET_FOLDER, scene_id, '{}_z.png').format(scene_id))
        if verbose:
            print('saving', os.path.join(DATASET_FOLDER, scene_id, '{}_z.png').format(scene_id))
    if visualize:
        plt.show()

    plt.close(fig)

def get_xy_relation_per_scene(dim, scene_id, visualize, savefig, dryrun=True, verbose=False, save_npy=False):

    if os.path.isfile(os.path.join(PROCESSED_DATA_FOLDER, '{}_{}.npy'.format(scene_id, 'y' if dim==1 else 'x'))) and save_npy:
        print('scene_id {} {} relation gt already existed. skipped!'.format(scene_id, 'y' if dim==1 else 'x'))
        return

    bboxes = np.load(os.path.join(PROCESSED_DATA_FOLDER, '{}_aligned_bbox.npy'.format(scene_id)))
    if not save_npy:
        bboxes = bboxes[bboxes[:, -1].argsort()]
    bboxes_min = bboxes[:, dim] - bboxes[:, dim+3]*0.5
    bboxes_max = bboxes[:, dim] + bboxes[:, dim+3]*0.5

    amax = bboxes_max[:, np.newaxis]
    bmax = np.tile(bboxes_max,(bboxes.shape[0],1))
    amin = bboxes_min[:, np.newaxis]
    bmin = np.tile(bboxes_min, (bboxes.shape[0], 1))
    bfirst = np.tile((bboxes_min + bboxes[:, dim+3]*0.3), (bboxes.shape[0],1))
    bsecond = np.tile((bboxes_min + bboxes[:, dim + 3] * 0.7), (bboxes.shape[0], 1))
    bepsilon = np.tile(bboxes[:, dim + 3] * 0.1, (bboxes.shape[0], 1))
    zero_mask = (abs(amax-bmax)<=bepsilon) & (abs(amin-bmin)<=bepsilon)
    forward_mask = ((amax>bmax) & (amin>=bmin)) | ((amax<=bmax) & (amax>bsecond) & (amin > bfirst)).astype(int)
    back_mask = (amax<bsecond)&(amin>bmin)&(amin<bfirst)
    forward_mask[(np.where(back_mask==1)[1], np.where(back_mask==1)[0])] = 1

    forward_mask[(np.where(forward_mask==1)[1], np.where(forward_mask==1)[0])] = -1
    forward_mask[(np.where(zero_mask == 1)[1], np.where(zero_mask == 1)[0])] = 0
    forward_mask[(np.where(zero_mask == 1)[0], np.where(zero_mask == 1)[1])] = 0


    if save_npy:
        out_npy = np.zeros(shape=forward_mask.shape, dtype=np.uint32) # 0: unannotated
        out_npy[forward_mask==0] = 1
        out_npy[forward_mask==-1] = 2
        out_npy[forward_mask==1] = 0
        if verbose:
            print(os.path.join(PROCESSED_DATA_FOLDER, '{}_{}.npy'.format(scene_id, 'y' if dim==1 else 'x')))
        if not dryrun:
            np.save(os.path.join(PROCESSED_DATA_FOLDER, '{}_{}.npy'.format(scene_id, 'y' if dim==1 else 'x')), out_npy)

        return



    json_f = json.load(open(os.path.join(DATASET_FOLDER, scene_id, '{}_obj.json'.format(scene_id))))
    tick_label = ['{}-{}'.format(bid, json_f[str(bid)]) for bid in bboxes[:,-1].astype(int)]
    df = pd.DataFrame(forward_mask, columns=tick_label,
                         index=tick_label)
    fig = plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    res = sns.heatmap(df, annot=True, vmin=forward_mask.min(), vmax=forward_mask.max(),
                      fmt='.2f', cmap=cmap, cbar_kws={"shrink": .82},
                      linewidths=0.1, linecolor='gray')

    if verbose:
        print("Relation along {}-axis for {}".format('Y' if dim == 1 else 'X', scene_id))
    plt.title("Relation along {}-axis for {}".format('Y' if dim == 1 else 'X', scene_id))

    if savefig:
        if not dryrun:
            plt.savefig(os.path.join(DATASET_FOLDER, scene_id, '{}_{}.png').format(scene_id, 'y' if dim==1 else 'x'))
        if verbose:
            print('saving', os.path.join(DATASET_FOLDER, scene_id, '{}_{}.png').format(scene_id, 'y' if dim == 1 else 'x'))

    if visualize:
        plt.show()

    plt.close(fig)


def generate_relation_gt_label(dataset, split, dryrun=True, verbose=True):
    get_scene_list(dataset)
    scn_list_p = os.path.join('./meta_data/{}_{}_scenelist.txt'.format(dataset, split))
    with open(scn_list_p, 'r') as f:
        scene_list = f.readlines()
    scene_list = [x.strip() for x in scene_list]
    print(dataset, split, len(scene_list))

    for scene_id in scene_list:
        get_z_relation_per_scene(scene_id=scene_id, visualize=False, savefig=False, dryrun=dryrun, verbose=verbose, save_npy=True)
        get_xy_relation_per_scene(dim=0, scene_id=scene_id, visualize=False, savefig=False, dryrun=dryrun, verbose=verbose, save_npy=True)
        get_xy_relation_per_scene(dim=1, scene_id=scene_id, visualize=False, savefig=False, dryrun=dryrun, verbose=verbose, save_npy=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="to visualize the spatiality relation label")
    parser.add_argument("--scene_id", type=str, help="the scene whose relation label would be visualized", default="scene0011_00")
    parser.add_argument("--axis", type=str, help="the relation along which axis, e.g. x|y|z", default="x")
    parser.add_argument("--savefig", action="store_true", help="to save the relation visualization under ./scans/scene_id/")

    parser.add_argument("--dataset", type=str, help="the target dataset to generate relation label: scanrefer|nr3d")
    parser.add_argument("--split", type=str, help="train|val")

    parser.add_argument("--verbose", action="store_true",
                        help="eval_visualize: print path info")
    parser.add_argument("--dryrun", action="store_true",
                        help="do not actually generate anything")
    args = parser.parse_args()
    dryrun, verbose = args.dryrun, args.verbose
    if args.visualize:
        scene_id, savefig = args.scene_id, args.savefig
        print('Visualizing spatiality relation along {}-axis for objects in {}'.format(args.axis, scene_id))
        get_obj_id_json_per_scene(scene_id=scene_id, dryrun=dryrun)
        if args.axis == 'x':
            get_xy_relation_per_scene(dim=0, scene_id=scene_id, visualize=True, savefig=savefig, dryrun=dryrun,
                                      verbose=verbose, save_npy=False)
        elif args.axis == 'y':
            get_xy_relation_per_scene(dim=1, scene_id=scene_id, visualize=True, savefig=savefig, dryrun=dryrun,
                                      verbose=verbose, save_npy=False)
        elif args.axis == 'z':
            get_z_relation_per_scene(scene_id=scene_id, visualize=True, savefig=savefig, dryrun=dryrun, verbose=verbose,
                                     save_npy=False)
        else:
            raise ValueError("Invalid axis.")
    else:
        dataset, split = args.dataset, args.split
        print('Generating spatiality relation labels for dataset {} {} split'.format(dataset, split))
        generate_relation_gt_label(dataset, split, dryrun=dryrun, verbose=verbose)


## to generate
## (dryrun) python generate_spatiality_label.py --dataset 'scanrefer' --split 'train' --verbose --dryrun
## (real) python generate_spatiality_label.py --dataset 'scanrefer' --split 'train' --verbose
## (dryrun) python generate_spatiality_label.py --dataset 'nr3d' --split 'train' --verbose --dryrun
## (real) python generate_spatiality_label.py --dataset 'nr3d' --split 'train' --verbose

## to visualize
## (dryrun) python generate_spatiality_label.py --visualize --scene_id 'scene0011_00' --axis x --verbose --dryrun
## (real) python generate_spatiality_label.py --visualize --scene_id 'scene0011_00' --axis x --verbose
## (save fig; dryrun) python generate_spatiality_label.py --visualize --scene_id 'scene0011_00' --axis x --savefig --verbose --dryrun
## (save fig; real) python generate_spatiality_label.py --visualize --scene_id 'scene0011_00' --axis z --savefig --verbose


