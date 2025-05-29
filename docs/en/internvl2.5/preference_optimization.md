# Mixed Preference Optimization

> Please use trl==0.10.1 to ensure the model works normally.

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

Before starting the preference optimization, download the pre-trained model we provide.

```sh
cd ckpt/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL2_5-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-8B --local-dir InternVL2_5-8B
# Download OpenGVLab/InternVL2_5-8B-MPO
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2_5-8B-MPO --local-dir InternVL2_5-8B-MPO
```

The directory structure is:

```sh
ckpt
├── InternVL2_5-8B
└── InternVL2_5-8B-MPO
```

## Prepare Our MMPR Dataset

To prepare the training data, please first download our [MMPR dataset](https://huggingface.co/datasets/OpenGVLab/MMPR-v1.1) and [the JSON file](https://huggingface.co/datasets/OpenGVLab/MMPR-v1.1/blob/main/meta.json).

Our dataset contains approximately 3 million preference pairs, of which only around 400k are utilized during training. You can adjust the number of active data samples and the data mixture ratio by modifying the `repeat` parameter in the JSON file.

The directory structure is:

```sh
MMPR
├── images
└── annotations
```

Please note that our training data includes instructions collected from [InternVL demo](https://internvl.opengvlab.com/). However, due to privacy protection concerns, we are unable to release these portion of the data.
Therefore, the reproduced results on general VQA (*i.e.*, MMVet, LLaVABench, and MMHal-Bench) may be inferior to [our released model](https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO).

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
  }
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
sh shell/internvl2.5_mpo/preference_optimization/internvl2_5_8b_internlm2_5_7b_dynamic_res_mpo.sh
```

If you encounter any issues, please let us know, and we will update the training guide to enhance its usability.

> Based on the environment of InternVL, you need to additionally run `pip install trl==0.10.1`.

## Evaluation

We evaluate the performance on other benchmarks (*e.g.*, MMVet, LLaVABench, and CRPE) using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). You need to set `use_mpo_prompt=True` in [config.py](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py) and `USE_COT="1"` in environment variable to activate the CoT prompt.

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
@article{chen2024expanding,
  title={Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  author={Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2412.05271},
  year={2024}
}
```

<br>
<br>
