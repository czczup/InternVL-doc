# Fine-tune on a Custom Dataset

## Model Preparation

| model name           | type | param | download                                                            |  size  |
| -------------------- | ---- | ----- | ------------------------------------------------------------------- | :----: |
| InternVL2-1B         | MLLM | 0.9B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-1B)         | 1.8 GB |
| InternVL2-2B         | MLLM | 2.2B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-2B)         | 4.2 GB |
| InternVL2-4B         | MLLM | 4.2B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-4B)         | 7.8 GB |
| InternVL2-8B         | MLLM | 8.1B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-8B)         | 16 GB  |
| InternVL2-26B        | MLLM | 25.5B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-26B)        | 48 GB  |
| InternVL2-40B        | MLLM | 40.1B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-40B)        | 75 GB  |
| InternVL2-Llama3-76B | MLLM | 76.3B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) | 143 GB |

Before starting the second fine-tuning, download the pre-trained model we provide.

```sh
cd pretrained/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL2-1B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-1B --local-dir InternVL2-1B
# Download OpenGVLab/InternVL2-2B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-2B --local-dir InternVL2-2B
# Download OpenGVLab/InternVL2-4B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-4B --local-dir InternVL2-4B
# Download OpenGVLab/InternVL2-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-8B --local-dir InternVL2-8B
# Download OpenGVLab/InternVL2-26B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-26B --local-dir InternVL2-26B
# Download OpenGVLab/InternVL2-40B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-40B --local-dir InternVL2-40B
# Download OpenGVLab/InternVL2-Llama3-76B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-Llama3-76B --local-dir InternVL2-Llama3-76B
```

The directory structure is:

```sh
pretrained
├── InternVL2-1B
├── InternVL2-2B
├── InternVL2-4B
├── InternVL2-8B
├── InternVL2-26B
├── InternVL2-40B
└── InternVL2-Llama3-76B
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

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2-1B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8x 32G/40G GPUs, whereas fine-tuning the LoRA requires 2x 32G/40G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 2B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2-2B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8x 32G/40G GPUs, whereas fine-tuning the LoRA requires 2x 32G/40G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 27G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 4B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2-4B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8x 40G GPUs, whereas fine-tuning the LoRA requires 2x 24G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 40G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 19G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 19G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_4b_phi3_3_8b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 8B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2-8B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 77G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 79G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 60G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 26B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_26b_internlm2_20b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_26b_internlm2_20b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2-26B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 77G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_26b_internlm2_20b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 79G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_26b_internlm2_20b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 60G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_26b_internlm2_20b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 40B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_40b_hermes2_yi_34b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_40b_hermes2_yi_34b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2-40B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 16 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 16 GPUs with SLURM system, fine-tune the full LLM, cost about 77G per GPU
PARTITION='your partition' GPUS=16 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_40b_hermes2_yi_34b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 74G per GPU
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_40b_hermes2_yi_34b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 74G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=2 sh shell/internvl2.0/2nd_finetune/internvl2_40b_hermes2_yi_34b_dynamic_res_2nd_finetune_lora.sh
```

````

````{tab} 76B

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_76b_hermes2_llama3_70b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_76b_hermes2_llama3_70b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL2-Llama3-76B`.

In the default settings, I have frozen the visual encoder. You can unfreeze it if needed. Generally, unfreezing the visual encoder will result in better performance.

> 💡 Fine-tuning the full LLM requires 32 A100 80G GPUs, whereas fine-tuning the LoRA requires 8 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 32 GPUs with SLURM system, fine-tune the full LLM, cost about 77G per GPU
PARTITION='your partition' GPUS=32 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_76b_hermes2_llama3_70b_dynamic_res_2nd_finetune_full.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 74G per GPU
GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_76b_hermes2_llama3_70b_dynamic_res_2nd_finetune_lora.sh
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
