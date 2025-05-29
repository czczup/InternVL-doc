# Evaluation of InternVL3 Series

To evaluate the performance of the InternVL3 series across various tasks, follow the instructions for each specific dataset. Ensure that the appropriate number of GPUs is allocated as specified.

> 1⃣️ We mainly use VLMEvalKit repositories for model evaluation.

> 2⃣️ Please note that evaluating the same model using different testing toolkits like InternVL and VLMEvalKit can result in slight differences, which is normal. Updates to code versions and variations in environment and hardware can also cause minor discrepancies in results.

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

Before evaluation, download the trained model we provide.

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

## Evaluation using VLMEvalKit Codebase

We evaluate the performance on most benchmarks (*e.g.*, MMVet, LLaVABench, and CRPE) using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). You need to set and `USE_COT="1"` in environment variable to activate the CoT prompt.

### Data Preparation

VLMEvalKit will automatically download the data for evaluation, so you do not need to prepare it manually.

### Evaluation on Different Benchmarks

To evaluate our models on different benchmarks, you can refer to the following script:

```sh
#!/bin/bash
set -x
PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-1}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

declare -a models=( \
  "InternVL3-1B" \
  "InternVL3-2B" \
  "InternVL3-8B" \
  "InternVL3-9B" \
  "InternVL3-14B" \
  "InternVL3-38B" \
  "InternVL3-78B" \
)

datasets="MMBench_TEST_EN_V11 MMStar MMMU_DEV_VAL MathVista_MINI HallusionBench AI2D_TEST OCRBench MMVet"
LOG_DIR="logs_eval"

export OPENAI_API_KEY="xxx"

for ((i=0; i<${#models[@]}; i++)); do

  model=${models[i]}

  if [[ "$model" =~ 38B|78B ]]; then
      GPUS_PER_TASK=8
  else
      GPUS_PER_TASK=1
  fi

  srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=$((GPUS / GPUS_PER_TASK)) \
    --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
    --quotatype=${QUOTA_TYPE} \
    --job-name="eval_wwy" \
    -o "${LOG_DIR}/${model}/evaluation.log" \
    -e "${LOG_DIR}/${model}/evaluation.log" \
    --async \
  python -u run.py \
    --data ${datasets} \
    --model ${model} \
    --verbose \

done
```

Note that VLMEvalkit does not officially support launching evaluation tasks with Slurm. You need to modify the [`run.py`](https://github.com/open-compass/VLMEvalKit/blob/main/run.py) script to support the Slurm launcher as follows:

```python
def init_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        pass
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.getenv('SLURM_PROCID', '0'))
        world_size = int(os.getenv('SLURM_NTASKS', '1'))
        local_rank = rank % torch.cuda.device_count()

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        if 'MASTER_ADDR' not in os.environ:
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            os.environ['MASTER_ADDR'] = addr
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '22110'

...

if __name__ == '__main__':
    load_env()
    init_dist()
    main()
```

Please refer to their [document](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) for more details.

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
