# Fine-tune on a Custom Dataset

## Model Preparation

| model name                 | type | param | download                                                                  |  size   |
| -------------------------- | ---- | ----- | ------------------------------------------------------------------------- | :-----: |
| InternVL-Chat-V1-5         | MLLM | 25.5B | 🤗 [HF link](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-V1-5)    | 48.0 GB |
| Mini-InternVL-Chat-2B-V1-5 | MLLM | 2.2B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5) | 4.2 GB  |
| Mini-InternVL-Chat-4B-V1-5 | MLLM | 4.2B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5) | 7.8 GB  |

Before starting the second fine-tuning, download the pre-trained model we provide.

```sh
cd pretrained/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL-Chat-V1-5
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-5 --local-dir InternVL-Chat-V1-5
# Download OpenGVLab/Mini-InternVL-Chat-2B-V1-5
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/Mini-InternVL-Chat-2B-V1-5 --local-dir Mini-InternVL-Chat-2B-V1-5
# Download OpenGVLab/Mini-InternVL-Chat-4B-V1-5
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/Mini-InternVL-Chat-4B-V1-5 --local-dir Mini-InternVL-Chat-4B-V1-5
```

The directory structure is:

```sh
pretrained
├── Mini-InternVL-Chat-2B-V1-5
├── Mini-InternVL-Chat-4B-V1-5
└── InternVL-Chat-V1-5
```

## Prepare Customized Data

After downloading the pre-trained model, prepare your customized SFT (Supervised Fine-Tuning) data. Create a JSON file in `internvl_chat/shell/data/` similar to [this example](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/data/internvl_1_2_finetune.json).

The format for the JSON file should be:

```json
{
  "your-custom-dataset-1": {
    "root": "path/to/the/image/",
    "annotation": "path/to/the/jsonl/annotation",
    "data_augment": false,
    "max_dynamic_patch": 12,
    "repeat_time": 1,
    "length": "number of samples in the dataset"
  }
}
```

Example:

```json
{
  "sharegpt4v_instruct_gpt4-vision_cap100k": {
    "root": "playground/data/",
    "annotation": "playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
    "data_augment": false,
    "max_dynamic_patch": 12,
    "repeat_time": 1,
    "length": 102025
  }
}
```

The format for each specific JSONL (such as plain text data, single-image data, multi-image data, video data) can be organized according to the descriptions provided in [this document](../get_started/chat_data_format.md).

My suggestion is to add new domain-specific data on top of the [general data from our open-sourced InternVL 1.2](../internvl1.2/reproduce.md#training-datasets-preparation). This will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

## Start 2nd Fine-tuning

`````{tabs}

````{tab} InternVL-Chat-V1-5

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL-Chat-V1-5`.

> 💡 Fine-tuning the full LLM requires 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 77G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 79G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 60G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} Mini-InternVL-Chat-2B-V1-5

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/Mini-InternVL-Chat-2B-V1-5`.

> 💡 Fine-tuning the full LLM requires 8x 32G/40G GPUs, whereas fine-tuning the LoRA requires 2x 32G/40G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} Mini-InternVL-Chat-4B-V1-5

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_phi3_3_8b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_phi3_3_8b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/Mini-InternVL-Chat-4B-V1-5`.

> 💡 Fine-tuning the full LLM requires 8x 40G GPUs, whereas fine-tuning the LoRA requires 2x 24G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 40G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_phi3_3_8b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 19G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_phi3_3_8b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 19G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_phi3_3_8b_dynamic_res_2nd_finetune_lora.sh
```

````

`````

If you encounter any issues, please let me know, and I will update the training guide to enhance its usability.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2024far,
  title={How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={Science China Information Sciences},
  volume={67},
  number={12},
  pages={220101},
  year={2024},
  publisher={Springer}
}
@inproceedings{chen2024internvl,
  title={Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24185--24198},
  year={2024}
}
```

<br>
<br>
