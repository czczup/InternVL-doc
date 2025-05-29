# Fine-tune on a Custom Dataset

## Model Preparation

| model name          | type | param | download                                                           |  size  |
| ------------------- | ---- | ----- | ------------------------------------------------------------------ | :----: |
| InternVL3-1B      | MLLM | 0.9B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL3-1B)      | 1.8 GB |
| InternVL3-2B  | MLLM | 2.1B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL3-2B)  | 4.2 GB |
| InternVL3-8B      | MLLM | 7.9B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL3-8B)      | 15.9 GB |
| InternVL3-9B  | MLLM | 9.1B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL3-9B)  | 18.3 GB |
| InternVL3-14B      | MLLM | 15.1B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL3-14B)      | 30.2 GB |
| InternVL3-38B  | MLLM | 38.4B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL3-38B)  | 76.8 GB |
| InternVL3-78B      | MLLM | 78.4B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL3-78B)      | 152 GB  |

Before starting the second fine-tuning, download the pre-trained model we provide.

```sh
pip install -U huggingface_hub

cd pretrained/
# Download OpenGVLab/InternVL3-1B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-1B --local-dir InternVL3-1B

# Download OpenGVLab/InternVL3-2B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-2B --local-dir InternVL3-2B

# Download OpenGVLab/InternVL3-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-8B --local-dir InternVL3-8B

# Download OpenGVLab/InternVL3-9B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-9B --local-dir InternVL3-9B

# Download OpenGVLab/InternVL3-14B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-14B --local-dir InternVL3-14B

# Download OpenGVLab/InternVL3-38B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-38B --local-dir InternVL3-38B

# Download OpenGVLab/InternVL3-78B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-78B --local-dir InternVL3-78B
```

The directory structure is:

```sh
pretrained
├── InternVL3-1B
├── InternVL3-2B
├── InternVL3-8B
├── InternVL3-9B
├── InternVL3-14B
├── InternVL3-38B
├── InternVL3-78B
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

````{tab} 1B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_1b_dynamic_res_2nd_finetune_full.sh).

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL3-1B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8x 32G/40G GPUs, whereas fine-tuning the LoRA requires 2x 32G/40G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl3.0/2nd_finetune/internvl3_1b_dynamic_res_2nd_finetune_full.sh
```

````

````{tab} 2B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_2b_dynamic_res_2nd_finetune_full.sh).

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL3-2B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8x 32G/40G GPUs, whereas fine-tuning the LoRA requires 2x 32G/40G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl3.0/2nd_finetune/internvl3_2b_dynamic_res_2nd_finetune_full.sh
```

````

````{tab} 8B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_full.sh).

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL3-8B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 40G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_full.sh
```

````

````{tab} 9B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_9b_dynamic_res_2nd_finetune_full.sh).

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL3-9B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 77G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl3.0/2nd_finetune/internvl3_9b_dynamic_res_2nd_finetune_full.sh
```

````

````{tab} 14B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_14b_dynamic_res_2nd_finetune_full.sh).

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL3-14B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 77G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl3.0/2nd_finetune/internvl3_14b_dynamic_res_2nd_finetune_full.sh
```

````

````{tab} 38B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_38b_dynamic_res_2nd_finetune_full.sh).

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL3-38B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 16 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

Commands for fine-tuning:

```sh
# Using 16 GPUs with SLURM system, fine-tune the full LLM, cost about 77G per GPU
PARTITION='your partition' GPUS=16 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl3.0/2nd_finetune/internvl3_38b_dynamic_res_2nd_finetune_full.sh
```

````

````{tab} 78B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl3.0/2nd_finetune/internvl3_78b_dynamic_res_2nd_finetune_full.sh).

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL3-78B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 32 A100 80G GPUs, whereas fine-tuning the LoRA requires 8 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 32 GPUs with SLURM system, fine-tune the full LLM, cost about 77G per GPU
PARTITION='your partition' GPUS=32 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl3.0/2nd_finetune/internvl3_78b_dynamic_res_2nd_finetune_full.sh
```

````

`````

If you encounter any issues, please let me know, and I will update the training guide to enhance its usability.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@misc{zhu2025internvl3exploringadvancedtraining,
      title={InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models}, 
      author={Jinguo Zhu and Weiyun Wang and Zhe Chen and Zhaoyang Liu and Shenglong Ye and Lixin Gu and Hao Tian and Yuchen Duan and Weijie Su and Jie Shao and Zhangwei Gao and Erfei Cui and Xuehui Wang and Yue Cao and Yangzhou Liu and Xingguang Wei and Hongjie Zhang and Haomin Wang and Weiye Xu and Hao Li and Jiahao Wang and Nianchen Deng and Songze Li and Yinan He and Tan Jiang and Jiapeng Luo and Yi Wang and Conghui He and Botian Shi and Xingcheng Zhang and Wenqi Shao and Junjun He and Yingtong Xiong and Wenwen Qu and Peng Sun and Penglong Jiao and Han Lv and Lijun Wu and Kaipeng Zhang and Huipeng Deng and Jiaye Ge and Kai Chen and Limin Wang and Min Dou and Lewei Lu and Xizhou Zhu and Tong Lu and Dahua Lin and Yu Qiao and Jifeng Dai and Wenhai Wang},
      year={2025},
      eprint={2504.10479},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.10479}, 
} 
```

<br>
<br>
