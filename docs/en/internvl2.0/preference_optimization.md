# Mixed Preference Optimization

## Model Preparation

| model name       | type | param | download                                                        | size  |
| ---------------- | ---- | ----- | --------------------------------------------------------------- | :---: |
| InternVL2-8B     | MLLM | 8.1B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-8B)     | 16 GB |
| InternVL2-8B-MPO | MLLM | 8.1B  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO) | 16 GB |

Before starting the preference optimization, download the pre-trained model we provide.

```sh
cd ckpt/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL2-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-8B --local-dir InternVL2-8B
# Download OpenGVLab/InternVL2-8B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-8B-MPO --local-dir InternVL2-8B-MPO
```

The directory structure is:

```sh
ckpt
├── InternVL2-8B
└── InternVL2-8B-MPO
```

## Prepare Our MMPR Dataset

To prepare the training data, please first download our [MMPR dataset](https://huggingface.co/datasets/OpenGVLab/MMPR) and [the JSON file](https://huggingface.co/datasets/OpenGVLab/MMPR/blob/main/meta.json).

Our dataset contains approximately 3 million preference pairs, of which only around 1.0 million are utilized during training. You can adjust the number of active data samples and the data mixture ratio by modifying the `repeat` parameter in the JSON file.

The directory structure is:

```sh
MMPR
├── images
└── annotations
```

Please note that our training data includes instructions collected from [InternVL demo](https://internvl.opengvlab.com/). However, due to privacy protection concerns, we are unable to release these portion of the data.
Therefore, the reproduced results on general VQA (*i.e.*, MMVet, LLaVABench, and MMHal-Bench) may be inferior to [our released model](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO).

We recommend incorporating additional general VQA data to preserve the general VQA abilities, following [our DropoutNTP pipeline](#generate-more-preference-data).

## Prepare Customized Data

If you want to prepare your customized preference data, please create a JSON file similar to [this example](https://huggingface.co/datasets/OpenGVLab/MMPR/blob/main/meta.json).

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
  "scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_format_rules": {
        "root": "MMPR/images/ScienceQA",
        "annotation": "MMPR/annotations/scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_format_rules.jsonl",
        "data_augment": false,
        "repeat_time": 1,
        "length": 66457
    },
}
```

The format for each specific JSONL (such as plain text data, single-image data, multi-image data) can be organized as the following format:

```json
{"image": "1.png", "question": "xxx", "chosen": "xxx", "rejected": "xxx",}
{"image": "2.png", "question": "xxx", "chosen": "xxx", "rejected": "xxx",}
...
```

Our suggestion is to add new domain-specific data on top of [MMPR](https://huggingface.co/datasets/OpenGVLab/MMPR). This will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

## Start Preference Optimization

Commands for preference optimization:

```sh
cd internvl_chat
sh shell/internvl2.0_mpo/preference_optimization/internvl2_8b_internlm2_7b_dynamic_res_mpo_full.sh
```

If you encounter any issues, please let us know, and we will update the training guide to enhance its usability.

> Based on the environment of InternVL, you need to additionally run `pip install trl==0.9.6`.

## Evaluation

To evaluate the resulting model with Chain-of-Though (CoT), please use the following commands:

```sh
# M3CoT
GPUS=8 sh evaluate.sh ckpt/InternVL2-8B-MPO m3cot --dynamic --cot
# MathVista
GPUS=8 sh evaluate.sh ckpt/InternVL2-8B-MPO mathvista-testmini --dynamic --cot
# POPE
GPUS=8 sh evaluate.sh ckpt/InternVL2-8B-MPO pope --dynamic --cot
```

Please note that we have organized the M3CoT data into the same format as ScienceQA. You can download the re-organized jsonl file [here](https://huggingface.co/datasets/Weiyun1025/M3CoT-ScienceQA-Format).

> We evaluate the performance on other benchmarks (*e.g.*, MMVet, LLaVABench, and CRPE) using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). You need to set `use_mpo_prompt=True` in [config.py](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py) to activate the CoT prompt.

## Generate Additional Preference Data

To construct additional open-ended VQA preference data, you can use our [DropoutNTP pipeline](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/tools/reasoning_data_pipeline/mmpr_data_pipeline_dropout_ntp.py) with the following command:

```shell
srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --nodes=${NODES} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
python -u tools/reasoning_data_pipeline/mmpr_data_pipeline_dropout_ntp.py \
    --checkpoint ${model_path} \  # the model you want to use to generate negative samples
    --prompt-path ${dataset} \  # please refer to the following format example
    --out-dir ${out_dir} \  # the output directory you want to save the resulting data
    --batch-size 1 \
    --num-workers 8 \
    --num-return-sequences 1 \  # the number of generated negative samples per item
    --top-k 50 \
    --temperature 1.0 \
    --dynamic \
    --max-num ${max_num} \  # max_tiles when enabling dynamic resolution
    --sample-max-num 500000 \
    --tp 8 \
    --start-ratio ${START_RATIO} \  # We set it to 0.5 by default
2>&1 | tee -a "${LOG_PATH}"  # the file path you want to save your log
```

The format for the prompt file should be:

```json
{"image": "1.png", "question": "xxx", "chosen": "xxx", "rejected": null,}
{"image": "2.png", "question": "xxx", "chosen": "xxx", "rejected": null,}
...
```

To constrct additional CoT reasoning preference data, you can use our [correctness-based pipeline](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/tools/reasoning_data_pipeline/mmpr_data_pipeline_correctness.py) with the following command:

```shell
srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --nodes=${NODES} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
python -u tools/reasoning_data_pipeline/mmpr_data_pipeline_correctness.py \
    --checkpoint ${model_path} \  # the model you want to use to generate negative samples
    --prompt-path ${dataset} \  # please refer to the following format example
    --out-dir ${out_dir} \  # the output directory you want to save the resulting data
    --batch-size 1 \
    --num-workers 8 \
    --num-return-sequences 32 \  # the number of generated reasoning processes per item
    --top-k 50 \
    --temperature 1.0 \
    --dynamic \
    --max-num ${max_num} \  # max_tiles when enabling dynamic resolution
    --sample-max-num 20000 \
    --tp 8 \
2>&1 | tee -a "${LOG_PATH}"  # the file path you want to save your log
```

The format for the prompt file should be:

```json
{"image": "1.png", "question": "xxx", "answer": "xxx"}
{"image": "2.png", "question": "xxx", "answer": "xxx"}
...
```

After sample multiple reasoning processes, you can use this command to convert them into preference data based on the correctness:

```shell
python -u tools/mm_reasoning_pipeline/internvl_lmdeploy_cot_postprocess.py \
    --data-dir "${data_dir}" \  # should be same with the ${out_dir} when sampling reasoning processes
    --save-dir "${save_dir}" \  # the output directory you want to save the resulting data
    --answer-fix \
    --force \
    --num-pairs-per-key 15 \
    --max-lines 1200000 \
```

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{wang2024mpo,
  title={Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization},
  author={Wang, Weiyun and Chen, Zhe and Wang, Wenhai and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Zhu, Jinguo and Zhu, Xizhou and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2411.10442},
  year={2024}
}
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
