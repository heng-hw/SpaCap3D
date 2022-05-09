# Spatiality-guided Transformer for 3D Dense Captioning on Point Clouds

Official implementation of "Spatiality-guided Transformer for 3D Dense Captioning on Point Clouds", IJCAI 2022. ([[arXiv]](https://arxiv.org/abs/2204.10688) [[project]](https://spacap3d.github.io/))

![teaser](docs/teaser.jpg)

updates:
* May 01, 2022: codes are released!

## Main Results

### ScanRefer

|Method | input | CIDEr<!-- -->@<!-- -->0.5IoU | BLEU-4<!-- -->@<!-- -->0.5IoU | METEOR<!-- -->@<!-- -->0.5IoU | ROUGE<!-- -->@<!-- -->0.5IoU | mAP<!-- -->@<!-- -->0.5IoU | Model | Eval. Cmd. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Scan2Cap](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scan2Cap_Context-Aware_Dense_Captioning_in_RGB-D_Scans_CVPR_2021_paper.html)| xyz| 32.94 | 20.63 | 21.10 | 41.58 | 27.45 | - | - | 
|[Scan2Cap](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scan2Cap_Context-Aware_Dense_Captioning_in_RGB-D_Scans_CVPR_2021_paper.html)| xyz+rgb+normal| 35.20 | 22.36 | 21.44 | 43.57 | 29.13 | - | - |
|[Scan2Cap](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scan2Cap_Context-Aware_Dense_Captioning_in_RGB-D_Scans_CVPR_2021_paper.html)| xyz+multiview+normal| 39.08 | 23.32 | 21.97 | 44.78 | 32.21 | - | - |
| Ours<sub>base</sub>| xyz |40.19 (38.61*) | 24.71 | 22.01 | 45.49 | 32.32 |[model](https://drive.google.com/drive/folders/1d2tHsPJDCBuWhbCOYSCCFzGBCInnM8JS?usp=sharing) | `python scripts/eval.py --eval_tag 'muleval'  --mul_eval --late_guide --no_learnt_src_pos --folder 'SPACAP_BASE'` |
| Ours| xyz |42.53 (40.47*) | 25.02 | 22.22 | 45.65 | 34.44 |[model](https://drive.google.com/drive/folders/1Yu43Wew6IryaSL-dKhIccPhUAM-elvG8?usp=sharing) | `python scripts/eval.py --eval_tag 'muleval'  --mul_eval --folder 'SPACAP'` |
| Ours| xyz+rgb+normal |42.76 (39.80*) | **25.38** | **22.84** |**45.66** | 35.55 |[model](https://drive.google.com/drive/folders/1Tb9aGv3yGLZzPw-2omrQZfRv7NFH6g4l?usp=sharing)| `python scripts/eval.py --eval_tag 'muleval'  --mul_eval --use_color --use_normal --folder 'SPACAP_RGB_NORMAL'` |
| Ours| xyz+multiview+normal |**44.02 (42.40\*)** |25.26 | 22.33 |45.36 | **36.64** |[model](https://drive.google.com/drive/folders/1dvMujPVO9B-0HWwDCQaKkB9xzOGT1RFo?usp=sharing)| `python scripts/eval.py --eval_tag 'muleval'  --mul_eval --use_multiview --use_normal --folder 'SPACAP_MV_NORMAL'` |

### Nr3D/ReferIt3D

|Method | input | CIDEr<!-- -->@<!-- -->0.5IoU | BLEU-4<!-- -->@<!-- -->0.5IoU | METEOR<!-- -->@<!-- -->0.5IoU | ROUGE<!-- -->@<!-- -->0.5IoU | mAP<!-- -->@<!-- -->0.5IoU | Model | Eval. Cmd. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[Scan2Cap](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Scan2Cap_Context-Aware_Dense_Captioning_in_RGB-D_Scans_CVPR_2021_paper.html)| xyz+multiview+normal| 24.10 | 15.01 | 21.01 | 47.95 | 32.21 | - | - |
| Ours<sub>base</sub>| xyz |31.06 (28.55*) | 17.94 | 22.03 | 49.63 | 30.65 |[model](https://drive.google.com/drive/folders/1pj4NGf4tPa4YSc923kMa-49hbjV_m8MI?usp=sharing) | `python scripts/eval.py --eval_tag 'muleval'  --mul_eval --late_guide --no_learnt_src_pos --folder 'SPACAP_BASE_NR3D'  --dataset ReferIt3D`|
| Ours| xyz |31.43 (29.35*) | 18.98 | 22.24 | 49.79 | 33.17 |[model](https://drive.google.com/drive/folders/1JxE9vKDDF1qcZm5WDQ8WpWA4D0M4OaYJ?usp=sharing) | `python scripts/eval.py --eval_tag 'muleval'  --mul_eval --folder 'SPACAP_NR3D' --dataset ReferIt3D` |
| Ours| xyz+rgb+normal |33.24 (31.01*) | 19.46 | **22.61** | 50.41 | 33.23 |[model](https://drive.google.com/drive/folders/1SpxnaHXFAu72RGqVurLqDGS5EZCUmhzw?usp=sharing) | `python scripts/eval.py --eval_tag 'muleval'  --mul_eval --use_color --use_normal --folder 'SPACAP_RGB_NORMAL_NR3D' --dataset ReferIt3D` |
| Ours| xyz+multiview+normal |**33.71 (30.52\*)** |**19.92** | 22.61 |**50.50** | **38.11** |[model](https://drive.google.com/drive/folders/1EHxF4WKjibLv2DgM7mhE7KCpzagIuTwB?usp=sharing) | `python scripts/eval.py --eval_tag 'muleval'  --mul_eval --use_multiview --use_normal --folder 'SPACAP_MV_NORMAL_NR3D' --dataset ReferIt3D` |

**Notes:**

- `*` means the CIDEr score is averaged over multiple evaluation as the algorithm randomness is large. The rest metrics are computed for the evaluation when the CIDEr score achieves the best.
- Ours<sub>base</sub>: standard encoder with sinusoidal positional encoding, late-guide decoder, and no token-to-token spatial relation guidance.
- Ours: token-to-token spatial relation guided encoder with learnable positional encoding, early-guide decoder
- To evaluate the model, put the downloaded model folder under `./outputs` as `./outputs/[--folder]/model.path`. And run the command in `Eval. Cmd.`. It would take ~4 hours.
- To download all the models at once, please click [here](https://drive.google.com/drive/folders/1T6by8nF395QoHLfFtikhwpJJgSpd0L3B?usp=sharing).
- All experiments were trained on a single GeForce RTX 2080Ti GPU.

## Installation

Please execute the following command to install PyTorch 1.6:

```shell
conda create -n spacap python=3.6.13
conda activate spacap
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2.89 -c pytorch
```


Install the necessary packages listed out in `requirements.txt`:
```shell
pip install -r requirements.txt
```

After all packages are properly installed, please run the following commands to compile the CUDA modules for the PointNet++ backbone:
```shell
cd lib/pointnet2
export CUDA_HOME=/usr/local/cuda-10.2
python setup.py install
```
__Before moving on to the next step, please don't forget to set the project root path to the `CONF.PATH.BASE` in `lib/config.py`.__


## Data Preparation
Download the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.

### ScanRefer
1. Download ScanRefer data [HERE](https://github.com/daveredrum/ScanRefer) and unzip it under `data/`.

2. Run `python scripts/organize_scanrefer.py` to generate organized ScanRefer `ScanRefer_filtered_organized.json` under `data/`. 

### ReferIt3D
1. Download ReferIt3D data (Nr3D only) [HERE](https://referit3d.github.io/#dataset) and put it under `data/`.

2. Run `python scripts/split_referit3d.py` to generate `nr3d_train.json` and `nr3d_val.json` under `data/`.

3. Run `python scripts/organize_referit3d.py` to generate organized Nr3D `nr3d_organized.json` under `data/`.

### ScanNet
In addition to ScanRefer and ReferIt3D, you also need to access the original ScanNet dataset to get the scene data. 

1. Follow instructions listed [HERE](./data/scannet/README.md) to get ScanNet data. After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`.

2. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command. Roughly 3.8GB free space is needed for this step:
    ```shell
    cd data/scannet/
    python batch_load_scannet_data.py
    ```
    
    > After this step, you can check if the processed scene data is valid by running:
    > ```shell
    > python visualize.py --scene_id scene0000_00
    > ```
    > Check the `*.obj` file under `/data/scannet/scannet_data`

3. To further generate axis-aligned mesh file `[scene_id]_axis_aligned.ply` under `data/scans/[scene_id]` for visualization:
    ```shell
    cd data/scannet/
    python align_axis.py
    ```

4. (optionally) To use 2D pretrained multiview feature as input:

    a. Download [the pretrained ENet weights (1.4MB)](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth) and put it under `data/`
    
    b. Download and unzip [the extracted ScanNet frames (~13GB)](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip) under `data/`.

    c. Extract the ENet features:
    ```shell
    python scripts/compute_multiview_features.py
    ```
        
    d. Project ENet features from ScanNet frames to point clouds; you need ~36GB to store the generated HDF5 database `enet_feats_maxpool.hdf5` under `data/scannet_data/`:
    ```shell
    python scripts/project_multiview_features.py --maxpool
    ```
    
    > You can check if the projections make sense by projecting the semantic labels from image to the target point cloud by:
    > ```shell
    > python scripts/project_multiview_labels.py --scene_id scene0000_00 --maxpool
    > ```
    > The projection would be saved under `/outputs/projections` as `scene0000_00.ply`.



### Relative Spatiality Label Generation
To equip the learning with ***token-to-token spatial relationship guidance***, we need to generate the ground truth spatiality labels for each scene from train split. The relative spatiality labels along three axes would be stored under `data/scannet/scannet_data` as `[scene_id]_x.npy`, `[scene_id]_y.npy`, and `[scene_id]_z.npy` after running the following scripts:
```
cd data/scannet/
python generate_spatiality_label.py --dataset 'scanrefer' --split 'train' --verbose
python generate_spatiality_label.py --dataset 'nr3d' --split 'train' --verbose
```

> You can also check if the relation label along `--axis` for a scene `--scene_id` is valid by visualizing:
> ```
> python generate_spatiality_label.py --visualize --scene_id 'scene0011_00' --axis x --savefig --verbose
> ```
> Note the `--savefig` flag saves the visualization as `./scans/[--scene_id]/[--scene_id]_[--axis].png`. Check [example](docs/scene0011_00_x.png).

__After data preparation, the dataset files are structured as follows.__
```
SpaCap
├── data
│   ├── ScanRefer_filtered_train.txt
│   ├── ScanRefer_filtered_val.txt
│   ├── ScanRefer_filtered.json 
│   ├── ScanRefer_filtered_train.json
│   ├── ScanRefer_filtered_val.json
│   ├── ScanRefer_filtered_organized.json
│   ├── nr3d.csv
│   ├── nr3d_train.json
│   ├── nr3d_val.json
│   ├── nr3d_organized.json 
│   ├── glove.p 
│   ├── scannet
│   │   ├── scans
│   │   │   ├── [scene_id]
│   │   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id].aggregation.json & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].txt & [scene_id]_axis_aligned.ply
│   │   ├── scannet_data
│   │   │   ├── enet_feats_maxpool.hdf5 (optional if you do not use --use_multiview)
│   │   │   ├── [scene_id]_aligned_bbox.npy & [scene_id]_aligned_vert.npy & [scene_id]_bbox.npy & [scene_id]_vert.npy & [scene_id]_ins_label.npy & [scene_id]_sem_label.npy & [scene_id]_x.npy & [scene_id]_y.npy & [scene_id]_z.npy

```


## Usage

### Training

To train our model with xyz as input (Training time: ~33h 22m):
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --tag 'spacap' --dataset 'ReferIt3D'
```

To train our model with xyz+rgb+normal as input (Training time: ~33h 47m):
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --tag 'spacap_rgb_normal' --use_color --use_normal
```

To train our model with xyz+multiview+normal as input (Training time: ~39h 40m):
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --tag 'spacap_mv_normal' --use_multiview --use_normal
```
> Note: the increased training time is mainly due to the fetch time of pretrained multiview features 

To train our base model Ours<sub>base</sub> with xyz as input (Training time: ~31h 14m):
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --tag 'spacap_base' --late_guide --no_relation --no_learnt_src_pos
```

Note if not specified, scripts above would train models on dataset ScanRefer by default. To train model on dataset ReferIt3D (Nr3D), toggle on flag `--dataset ReferIt3D`.

The trained model as well as the intermediate results will be dumped into `outputs/<output_folder>` where `<output_folder>` would be `timestamp_[--tag]` (e.g, `2022-04-20_11-59-59_SPACAP`). 

### Evaluation
For evaluating the model (@0.5IoU) multiple times to find the best performance in CIDEr score, please run the following script and change the `<output_folder>` accordingly:
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py --eval_tag 'muleval'  --mul_eval --folder <output_folder> 
```

Specific evaluation scripts for different model setting are provided in [Eval. Cmd.](#main-results).

### Visualization
To visualize the predicted bounding box and caption for each object, please run the following script:
```shell
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py --eval_tag 'vis' --seed 25 --eval_visualize --folder <output_folder> --nodryrun
```
A folder `/vis` would be created under `/outputs/<output_folder>` where predicted caption for each testing scene would be saved as `scene_id/predictions.json` under `/vis` and the object bounding box prediction would be saved as `scene_id/pred-[obj_id]-[obj_name].ply`.

> Note the `--seed` can be any number or the one when your model achieves the highest CIDEr score. 

## Citation
If you find our work helpful in your research, please kindly cite our paper via:
```bibtex
@inproceedings{SpaCap3D,
    title={Spatiality-guided Transformer for 3{D} Dense Captioning on Point Clouds},
    author={Wang, Heng and Zhang, Chaoyi and Yu, Jianhui and Cai, Weidong},
    booktitle={Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
    year={2022}
}
```

## Acknowledgement
This repo is built mainly upon [Scan2Cap](https://github.com/daveredrum/Scan2Cap). We also borrow code from [annotated-transformer](https://github.com/harvardnlp/annotated-transformer) for the basic Transformer building blocks.

## Contact
If you have any questions or suggestions about this repo, please feel free to contact me! ([heng.wang@sydney.edu.au](heng.wang@sydney.edu.au))

