# InternViT-6B for Segmentation

> **Version: InternVL 1.0.**

This code corresponds to Section 4.2.2 of our [InternVL 1.0 paper](https://arxiv.org/pdf/2312.14238).

In this part, we validate the visual perception capabilities of InternViT-6B, the most core component of InternVL 1.0.
To investigate the pixel-level perceptual capacity of InternViT-6B, we conduct extensive experiments of semantic segmentation on the ADE20K dataset.

## Data Preparation

To set up your dataset for segmentation, it is recommended to symlink the dataset root to `segmentation/data`. If your folder structure is different, you may need to adjust the corresponding paths in the config files.
  
```none
segmentation
├── data
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
```

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).

If you want to use other datasets, please refer to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## Model Preparation

| model name         | type    | download                                                                                  | size  |
| ------------------ | ------- | ----------------------------------------------------------------------------------------- | :---: |
| InternViT-6B-224px | pytorch | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_224px.pth) | 12 GB |

Please download the above model weight and place it in the `pretrained/` folder:

```shell
mkdir pretrained && cd pretrained
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_224px.pth
```

The directory structure should be:

```sh
pretrained
└── intern_vit_6b_224px.pth
```

## Training

> Note: This open-source code does not include DeepSpeed in MMSegmentation. It currently supports training for linear probing and head tuning only, and does not support full-parameter training.

To train a linear classifier for `InternViT-6B` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py 8
# or manage jobs with slurm
GPUS=8 sh slurm_train.sh <partition> <job-name> configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py
```

## Evaluation

| type            | backbone              |  head   | mIoU |                                                                                config                                                                                 |                                                                                                                      download                                                                                                                       |
| --------------- | --------------------- | :-----: | :--: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| few-shot (1/16) | InternViT-6B          | Linear  | 46.5 |     [config](https://github.com/OpenGVLab/InternVL/blob/main/segmentation/configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_5k_ade20k_bs16_lr4e-5_1of16.py)     |    [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_5k_ade20k_bs16_lr4e-5_1of16.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_5k_ade20k_bs16_lr4e-5_1of16.log)    |
| few-shot (1/8)  | InternViT-6B          | Linear  | 50.0 |     [config](https://github.com/OpenGVLab/InternVL/blob/main/segmentation/configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_10k_ade20k_bs16_lr4e-5_1of8.py)     |    [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_10k_ade20k_bs16_lr4e-5_1of8.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_10k_ade20k_bs16_lr4e-5_1of8.log)    |
| few-shot (1/4)  | InternViT-6B          | Linear  | 53.3 |     [config](https://github.com/OpenGVLab/InternVL/blob/main/segmentation/configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_20k_ade20k_bs16_lr4e-5_1of4.py)     |    [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_20k_ade20k_bs16_lr4e-5_1of4.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_20k_ade20k_bs16_lr4e-5_1of4.log)    |
| few-shot (1/2)  | InternViT-6B          | Linear  | 55.8 |     [config](https://github.com/OpenGVLab/InternVL/blob/main/segmentation/configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_40k_ade20k_bs16_lr4e-5_1of2.py)     |    [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_40k_ade20k_bs16_lr4e-5_1of2.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_40k_ade20k_bs16_lr4e-5_1of2.log)    |
| few-shot (1/1)  | InternViT-6B          | Linear  | 57.2 |     [config](https://github.com/OpenGVLab/InternVL/blob/main/segmentation/configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_1of1.py)     |    [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_1of1.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_1of1.log)    |
| linear probing  | InternViT-6B (frozen) | Linear  | 47.2 | [config](https://github.com/OpenGVLab/InternVL/blob/main/segmentation/configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py) |  [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.log)  |
| head tuning     | InternViT-6B (frozen) | UperNet | 54.9 |  [config](https://github.com/OpenGVLab/InternVL/blob/main/segmentation/configs/intern_vit_6b/head_tuning/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py)  | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.log) |
| full tuning     | InternViT-6B          | UperNet | 58.9 |     [config](https://github.com/OpenGVLab/InternVL/blob/main/segmentation/configs/intern_vit_6b/full_tuning/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.py)      |        [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.log)        |

You can download checkpoints from [here](https://huggingface.co/OpenGVLab/InternVL/tree/main) or from the table above. Then place them to `segmentation/checkpoints/`.

For example, to evaluate a model with a single GPU:

```bash
python test.py configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py checkpoints/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth --eval mIoU
```

For example, to evaluate a model with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py checkpoints/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth 8 --eval mIoU
```
