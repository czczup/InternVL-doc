# Fine-tune on a Custom Dataset

## Model Preparation

| model name          | type | param | download                                                           |  size  |
| ------------------- | ---- | ----- | ------------------------------------------------------------------ | :----: |
| InternVL2_5-1B      | MLLM | 0.9B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-1B)      | 1.8 GB |
| InternVL2_5-1B-MPO  | MLLM | 0.9B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-1B-MPO)  | 1.8 GB |
| InternVL2_5-2B      | MLLM | 2.2B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-2B)      | 4.2 GB |
| InternVL2_5-2B-MPO  | MLLM | 2.2B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-2B-MPO)  | 4.2 GB |
| InternVL2_5-4B      | MLLM | 4.2B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-4B)      | 7.8 GB |
| InternVL2_5-4B-MPO  | MLLM | 4.2B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-4B-MPO)  | 7.8 GB |
| InternVL2_5-8B      | MLLM | 8.1B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-8B)      | 16 GB  |
| InternVL2_5-8B-MPO  | MLLM | 8.1B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO)  | 16 GB  |
| InternVL2_5-26B     | MLLM | 25.5B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-26B)     | 48 GB  |
| InternVL2_5-26B-MPO | MLLM | 25.5B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-26B-MPO) | 48 GB  |
| InternVL2_5-38B     | MLLM | 40.1B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-38B)     | 75 GB  |
| InternVL2_5-38B-MPO | MLLM | 40.1B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-38B-MPO) | 75 GB  |
| InternVL2_5-78B     | MLLM | 76.3B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-78B)     | 143 GB |
| InternVL2_5-78B-MPO | MLLM | 76.3B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2_5-78B-MPO) | 143 GB |

Before starting the second fine-tuning, download the pre-trained model we provide.

```sh
cd pretrained/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL2_5-1B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-1B --local-dir InternVL2_5-1B

# Download OpenGVLab/InternVL2_5-1B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-1B-MPO --local-dir InternVL2_5-1B-MPO

# Download OpenGVLab/InternVL2_5-2B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-2B --local-dir InternVL2_5-2B

# Download OpenGVLab/InternVL2_5-2B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-2B-MPO --local-dir InternVL2_5-2B-MPO

# Download OpenGVLab/InternVL2_5-4B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-4B --local-dir InternVL2_5-4B

# Download OpenGVLab/InternVL2_5-4B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-4B-MPO --local-dir InternVL2_5-4B-MPO

# Download OpenGVLab/InternVL2_5-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-8B --local-dir InternVL2_5-8B

# Download OpenGVLab/InternVL2_5-8B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-8B-MPO --local-dir InternVL2_5-8B-MPO

# Download OpenGVLab/InternVL2_5-26B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-26B --local-dir InternVL2_5-26B

# Download OpenGVLab/InternVL2_5-26B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-26B-MPO --local-dir InternVL2_5-26B-MPO

# Download OpenGVLab/InternVL2_5-38B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-38B --local-dir InternVL2_5-38B

# Download OpenGVLab/InternVL2_5-38B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-38B-MPO --local-dir InternVL2_5-38B-MPO

# Download OpenGVLab/InternVL2_5-78B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-78B --local-dir InternVL2_5-78B

# Download OpenGVLab/InternVL2_5-78B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-78B-MPO --local-dir InternVL2_5-78B-MPO
```

The directory structure is:

```sh
pretrained
├── InternVL2_5-1B
├── InternVL2_5-1B-MPO
├── InternVL2_5-2B
├── InternVL2_5-2B-MPO
├── InternVL2_5-4B
├── InternVL2_5-4B-MPO
├── InternVL2_5-8B
├── InternVL2_5-8B-MPO
├── InternVL2_5-26B
├── InternVL2_5-26B-MPO
├── InternVL2_5-38B
├── InternVL2_5-38B-MPO
├── InternVL2_5-78B
└── InternVL2_5-78B-MPO
```

## Prepare Your Customized Training Data

After downloading the pre-trained model, prepare your customized SFT (Supervised Fine-Tuning) data. Create a JSON file in `internvl_chat/shell/data/` similar to [this example](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/data/internvl_1_2_finetune.json).

The format for the JSON file should be:

```json
{
  "your-custom-dataset-1": {
    "root": "path/to/the/image/",
    "annotation": "path/to/the/jsonl/annotation",
    "data_augment": false,
    "repeat_time": 1,
    "length": "number of samples in the dataset"
  },
  ...
}
```

Example:

```json
{
  "sharegpt4v_instruct_gpt4-vision_cap100k": {
    "root": "playground/data/",
    "annotation": "playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
    "data_augment": false,
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

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2_5-1B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8x 32G/40G GPUs, whereas fine-tuning the LoRA requires 2x 32G/40G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_1b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 2B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_2b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_2b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2_5-2B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8x 32G/40G GPUs, whereas fine-tuning the LoRA requires 2x 32G/40G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_2b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_2b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_2b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 4B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_4b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_4b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2_5-4B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8x 40G GPUs, whereas fine-tuning the LoRA requires 2x 24G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 40G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_4b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 19G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_4b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 19G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_4b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 8B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_8b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_8b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2_5-8B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 77G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_8b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 79G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_8b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 60G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_8b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 26B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_26b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_26b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2_5-26B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 77G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_26b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 79G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_26b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 60G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_26b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 38B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_38b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_38b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2_5-38B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 16 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 16 GPUs with SLURM system, fine-tune the full LLM, cost about 77G per GPU
PARTITION='your partition' GPUS=16 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_38b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 74G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_38b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 74G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.5/2nd_finetune/internvl2_5_38b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 78B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_78b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.5/2nd_finetune/internvl2_5_78b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2_5-78B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 32 A100 80G GPUs, whereas fine-tuning the LoRA requires 8 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 32 GPUs with SLURM system, fine-tune the full LLM, cost about 77G per GPU
PARTITION='your partition' GPUS=32 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.5/2nd_finetune/internvl2_5_78b_dynamic_res_2nd_finetune_full.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 74G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh internvl2.5/2nd_finetune/internvl2_5_78b_dynamic_res_2nd_finetune_lora.sh
```

````

`````

If you encounter any issues, please let me know, and I will update the training guide to enhance its usability.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2024expanding,
  title={Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  author={Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2412.05271},
  year={2024}
}
@article{wang2024mpo,
  title={Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization},
  author={Wang, Weiyun and Chen, Zhe and Wang, Wenhai and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Zhu, Jinguo and Zhu, Xizhou and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2411.10442},
  year={2024}
}
@article{gao2024mini,
  title={Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5\% Parameters and 90\% Performance},
  author={Gao, Zhangwei and Chen, Zhe and Cui, Erfei and Ren, Yiming and Wang, Weiyun and Zhu, Jinguo and Tian, Hao and Ye, Shenglong and He, Junjun and Zhu, Xizhou and others},
  journal={arXiv preprint arXiv:2410.16261},
  year={2024}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
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
