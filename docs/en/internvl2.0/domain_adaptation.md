# Domain Adaptation

## Multi-View Image-Based Autonomous Driving

### Data Preparation

- Prepare *InternVL-Chat-V1-2-SFT-Data*, See [Document](../internvl1.2/reproduce.md/#training-datasets-preparation)
- Download `drivelm_train.jsonl` and `drivelm_val.jsonl` from[InternVL-Domain-Adaptation-Data](https://huggingface.co/datasets/OpenGVLab/InternVL-Domain-Adaptation-Data). `drivelm_train.jsonl` and `drivelm_val.jsonl` are the data after format conversion.

- Download the images from [DriveLM](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge) and process the images using `tools/images_stitching.py`:

```sh
python tools/images_stitching.py --data-root InternVL-Domain-Adaptation-Data/images/drivelm --ann-file path/to/v1_1_val_nus_q_only.json
```

- Download autonomous driving subset of [mme-realworld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld/tree/main).

- Organize the files according to the following structure.

  ```
  path/to/internvl_chat/InternVL-Domain-Adaptation-Data 
  ├── train_data
  │   └── drivelm_train.jsonl
  ├── images
  │   ├── MME-RealWorld
  |   |   └── data/AutonomousDriving/
  |   └── drivelm
  |       ├── nuscenes/
  |       └── stitch/
  ├── train_meta
  |   ├── internvl_1_2_finetune_drivelm.json
  └── val
      ├── MME_RealWorld.json
      └── drivelm_val.jsonl
  ```

### Finetune

After downloading the pre-trained model and preparing the training data, you can adapte the model using following scripts.

Before fine-tuning, set the `--model_name_or_path` to the path of the path of the pre-trained model.

In the default settings, we conduct full-parameter fine-tuning, but you can optionally freeze the visual encoder depending on your computational resources.

`````{tabs}

````{tab} Mini-InternVL-1B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_1b_qwen2_0_5b_dynamic_res_finetune_drivelm.sh
```

````

````{tab} Mini-InternVL-2B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_2b_internlm2_1_8b_dynamic_res_finetune_drivelm.sh
```

````
````{tab} Mini-InternVL-4B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_4b_phi3_3_8b_dynamic_res_finetune_drivelm.sh
```

````

`````

### Evaluation

- [DriveLM Challenge](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge)

This dataset contains data for perception, prediction, and planning, providing a comprehensive view of autonomous driving scenarios. To test our fine-tuned model on the DriveLM Challenge, we have already pre-processed the data, including both images and annotations. You can now directly use the following command to run the test with 8 GPUs:

```bash
GPUS=8 sh evaluate.sh ${checkpoint} drivelm
```

- MME-Realworld-AD

[MME-Realworld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld/tree/main) contains a subset of autonomous driving scenes, on which we assess the model's performance on *perception* and *reasoning* tasks.
Please use the following command to perform the test with 8 GPU:

```bash
GPUS=8 sh evaluate.sh ${checkpoint} mme—realworld --dynamic --max-num  12 --subtask  Autonomous_Driving
```

## Medical Images

### Data Preparation

- Prepare *InternVL-Chat-V1-2-SFT-Data*, See [Document](../internvl1.2/reproduce.md/#training-datasets-preparation)

- Download the following files from[InternVL-Domain-Adaptation-Data](https://huggingface.co/datasets/OpenGVLab/InternVL-Domain-Adaptation-Data), extract the images, and organize them into the following directory structure.

```
path/to/internvl_chat/InternVL-Domain-Adaptation-Data 
├── train_data
│   └── medical_sft_sample500k.jsonl
├── images
│   └── medical_images
└── train_meta
    └── internvl_1_2_finetune_medical.json
```

### Finetune

Please finetune the model using following scripts:

`````{tabs}

````{tab} 1B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_1b_qwen2_0_5b_dynamic_res_finetune_medical.sh

````

````{tab} 2B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_2b_internlm2_1_8b_dynamic_res_finetune_medical.sh
```

````
````{tab} 4B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_4b_phi3_3_8b_dynamic_res_finetune_medical.sh
```

````

`````

### Evaluation

we test our model on a comprehensive medical AI benchmark,
[GMAI-MMBench](https://github.com/uni-medical/GMAI-MMBench). Our
evaluation was conducted using the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) framework.

Please refer to [Document](https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench) for testing.

Importantly, before testing, please add the model to the `internvl_series` in [config_file](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py):

```
  'Mini-InternVL-DA-1B': partial(InternVLChat, model_path='path/to/your/checkpoints', version='V2.0'),
  'Mini-InternVL-DA-2B': partial(InternVLChat, model_path='path/to/your/checkpoints', version='V2.0'),
  'Mini-InternVL-DA-4B': partial(InternVLChat, model_path='path/to/your/checkpoints', version='V2.0')
```

## Remote Sensing

### Data Preparation

- Prepare *InternVL-Chat-V1-2-SFT-Data*, See [Document](../internvl1.2/reproduce.md/#training-datasets-preparation)
- Please download the corresponding files in train_data, train_meta, and val directories from [InternVL-Domain-Adaptation-Data](https://huggingface.co/datasets/OpenGVLab/InternVL-Domain-Adaptation-Data), following the directory tree structure below.

- Download the images from [GeoChat](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct/tree/main), [FIT-RS](https://huggingface.co/datasets/ll-13/FIT-RS/blob/main/FIT-RS_Instruction/FIT-RS_Img.tar.gz), [RSVQA](https://rsvqa.sylvainlobry.com/) and [DIOR-RSVG](https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_). Extract the files and place them in the corresponding locations within the directory structure below.

```
path/to/internvl_chat/InternVL-Domain-Adaptation-Data 
├── train_data
│   ├── dior_rsvg_instruct_26k.jsonl
|   ├── fit_rs_vqa_100k.jsonl
|   ├── rsvqa_hr_train_instruct_100k.jsonl
│   └── geochat_instruct.jsonl
├── images
|   ├── RSVQA_L
|   |   └── Images_LR
|   ├── RSVQA-H
|   |   └── Data
|   ├── DIOR-RSVG
|   |   └── JPEGImages
|   ├── FIT-RS
|   |   └── imgv2_split_512_100_vaild
|   └── GeoChat
|       └── images
|           └── final_images_llava
├── train_meta
|   └── internvl_1_2_finetune_remote.json
└── val
    ├── dior_rsvg_test.json
    ├── rsvqa_h_test_1_instruct.json
    ├── rsvqa_h_test_2_instruct.json
    └── rsvqa_l_test_instruct.json
```
### Finetune

Please finetune the model using following scripts:

`````{tabs}

````{tab} 1B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_1b_qwen2_0_5b_dynamic_res_finetune_remote.sh

````

````{tab} 2B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_2b_internlm2_1_8b_dynamic_res_finetune_remote.sh
```

````
````{tab} 4B

```sh
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/mini_internvl/domain_adaptation/internvl2_4b_phi3_3_8b_dynamic_res_finetune_remote.sh
```

`````

### Evaluation

We assess the performance of our transferred model using the RSVQA dataset for the VQA task and the DIOR-RSVG dataset for the visual grounding task.

- RS-VQA

We chose the Presence, Comparison, and Rural/Urban subsets of the RSVQA-LR and RSVQA-HR datasets for assessment.

You can now directly use the following command to run the test with 8 GPUs:
```bash
# RSVQA-LR 
GPUS=8 sh evaluate.sh ${checkpoint} rsvqa-lr --dynamic --max-num  12
# RSVQA-HR-test1
GPUS=8 sh evaluate.sh ${checkpoint} rsvqa-hr-test1 --dynamic --max-num  12
# RSVQA-LR-test2
GPUS=8 sh evaluate.sh ${checkpoint} rsvqa-hr-test2 --dynamic --max-num  12
```

- DIOR-RSVG

You can now directly use the following command to run the test with 8 GPUs:

```bash
GPUS=8 sh evaluate.sh ${checkpoint} dior-rsvg --dynamic --max-num  12
```

## Autonomous Driving with Temporal Information

Coming soon...

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

<br>
<br>
