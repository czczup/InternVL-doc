# Evaluation of InternVL2.5 Series

To evaluate the performance of the InternVL2.5 series across various tasks, follow the instructions for each specific dataset. Ensure that the appropriate number of GPUs is allocated as specified.

> 1⃣️ We mainly use VLMEvalKit repositories for model evaluation.

> 2⃣️ Please note that evaluating the same model using different testing toolkits like InternVL and VLMEvalKit can result in slight differences, which is normal. Updates to code versions and variations in environment and hardware can also cause minor discrepancies in results.

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

Before evaluation, download the trained model we provide.

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
  "InternVL2-5-1B" \
  "InternVL2-5-1B-MPO" \
  "InternVL2-5-2B" \
  "InternVL2-5-2B-MPO" \
  "InternVL2-5-4B" \
  "InternVL2-5-4B-MPO" \
  "InternVL2-5-8B" \
  "InternVL2-5-8B-MPO" \
  "InternVL2-5-38B" \
  "InternVL2-5-38B-MPO" \
  "InternVL2-5-78B" \
  "InternVL2-5-78B-MPO" \
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
