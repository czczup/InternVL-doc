# Mixed Preference Optimization

> Please use trl==0.10.1 to ensure the model works normally.

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

Before starting the preference optimization, download the pre-trained model we provide.

```sh
cd ckpt/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL3-8B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-8B --local-dir InternVL3-8B
```

The directory structure is:

```sh
ckpt
├── InternVL3-8B
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
Therefore, the reproduced results on general VQA (*i.e.*, MMVet, LLaVABench, and MMHal-Bench) may be inferior to [our released model](https://huggingface.co/OpenGVLab/InternVL3-8B).

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
sh shell/internvl3.0/mpo/internvl3_8b_mpo.sh
```

If you encounter any issues, please let us know, and we will update the training guide to enhance its usability.

> Based on the environment of InternVL, you need to additionally run `pip install trl==0.10.1`.

## Evaluation

We evaluate the performance on other benchmarks (*e.g.*, MMVet, LLaVABench, and CRPE) using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). You need to set `use_mpo_prompt=True` in [config.py](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py) and `USE_COT="1"` in environment variable to activate the CoT prompt.

## Generate Additional Preference Data

To construct additional open-ended VQA preference data, you can use our [DropoutNTP pipeline](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/tools/reasoning_data_pipeline/mmpr_data_pipeline_dropout_ntp.py) with the following command:

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

To constrct additional CoT reasoning preference data, you can use our [correctness-based pipeline](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/tools/reasoning_data_pipeline/mmpr_data_pipeline_correctness.py) with the following command:

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
python -u tools/reasoning_data_pipeline/mmpr_data_pipeline_correctness_postprocess.py \
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
