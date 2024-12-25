# Introduction of InternVL2.5 Series

We introduce InternVL2.5, an advanced multimodal large language model (MLLM) series that builds upon InternVL 2.0, maintaining its core model architecture while introducing significant enhancements in training and testing strategies as well as data quality.

In this work, we delve into the relationship between model scaling and performance, systematically exploring the performance trends in vision encoders, language models, dataset sizes, and test-time configurations. Through extensive evaluations on a wide range of benchmarks, InternVL2.5 exhibits competitive performance, rivaling leading commercial models such as GPT-4o and Claude-3.5-Sonnet.

Notably, **our model is the first open-source MLLMs to achieve over 70% on the MMMU benchmark.** We hope this model contributes to the open-source community by setting new standards for developing and applying multimodal AI systems.

![image](./internvl2.5.jpg)

InternVL2.5 family is built upon the following designs:

- **Progressive Scaling Strategy**: We propose a progressive scaling strategy to efficiently align the vision encoder with LLMs. This strategy adopts a staged training approach, starting with smaller, resource-efficient LLMs and progressively scaling up to larger LLMs. This approach stems from our observation that even when the ViT and LLM are jointly trained using NTP loss, the resulting visual features are generalizable representations that can be easily understood by other LLMs. Specifically, the InternViT is trained alongside a smaller LLM (e.g., 20B), focusing on optimizing fundamental visual capabilities and cross-modal alignment. This phase avoids the high computational costs associated with training directly with a large LLM. Using a shared-weight mechanism, the trained InternViT can be seamlessly transferred to a larger LLM (e.g., 72B) without requiring retraining. Consequently, when training a larger model, much less data is required and the computation cost is significantly reduced.

- **Improved Training Strategy**: To enhance the model’s adaptability to real-world scenarios and overall performance, we introduce two key techniques: `Random JPEG Compression` and `Loss Reweighting`. For Random JPEG Compression, random JPEG compression with quality levels between 75 and 100 is applied to simulate the degradation commonly found in internet-sourced images. For Loss Reweighting, we express the widely applied strategies (i.e., token averaging and sample averaging) in a unified format and propose square averaging to balance the gradients biases towards long or short responses.

- **Well-structed Data Organization**: During model development, we observed that even a small fraction of anomalous samples can lead to aberrant model behavior during inference. To address this issue, we propose a filtering pipeline consisting of LLM-Based Quality Scoring and Rule-Based Filtering, which significantly reduced the occurrence of anomalous behaviors, particularly repetitive generation, with notable improvements in CoT reasoning tasks. Additionally, we implement a data-packing strategy to enhance GPU utilization and improve training efficiency.

![image](./arch.png)

As shown in this figure, InternVL2.5 utilizes the same architecture as InternVL 1.5 and InternVL 2.0, specifically the `ViT-MLP-LLM` configuration referenced in various existing MLLM studies.
For the various sizes of the InternVL2.5 model, we employed different visual encoders and large language models, as detailed in the table below.

|   Model Name    |                                       Vision Part                                       |                                 Language Part                                  |                           HF Link                           |                                MS Link                                 |
| :-------------: | :-------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :---------------------------------------------------------: | :--------------------------------------------------------------------: |
| InternVL2_5-1B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |   [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-1B)  | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-1B)  |
| InternVL2_5-2B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) | [internlm2_5-1_8b-chat](https://huggingface.co/internlm/internlm2_5-1_8b-chat) | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-2B)  | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-2B)  |
| InternVL2_5-4B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |     [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)     | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-4B)  | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-4B)  |
| InternVL2_5-8B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |   [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat)   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-8B)  | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-8B)  |
| InternVL2_5-26B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |  [internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-26B) | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-26B) |
| InternVL2_5-38B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |    [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-38B) | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-38B) |
| InternVL2_5-78B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |    [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-78B) | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-78B) |

## Mixed Preference Optimization

We also introduce InternVL2.5-MPO, which is fine-tuned with Mixed Preference Optimization (MPO). **These models outperform their counterparts without MPO by an average of 2 points across all scales on the OpenCompass leaderboard.**

InternVL2.5-MPO family is built upon the following designs:

- [**Multi-Modal Preference Dataset (MMPR)**](https://huggingface.co/datasets/OpenGVLab/MMPR-v1.1): We propose an efficient preference data construction pipeline. Based on this pipeline, we create MMPR, a high-quality, large-scale multimodal reasoning preference dataset containing approximately 3 million samples.
- [**Mixed Preference Optimization (MPO)**](https://huggingface.co/collections/OpenGVLab/internvl25-mpo-6753fed98cd828219b12f849): We introduce MPO, an effective PO algorithm designed to improve the reasoning abilities of MLLMs. The key insight behind this algorithm is that an effective PO process should enable the model to learn the relative preference between pairs of responses, the absolute quality of individual responses, and the process for generating preferred responses.

|     Model Name      |                                       Vision Part                                       |                                 Language Part                                  |                             HF Link                             |                                  MS Link                                   |
| :-----------------: | :-------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :-------------------------------------------------------------: | :------------------------------------------------------------------------: |
| InternVL2_5-1B-MPO  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |   [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-1B-MPO)  | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-1B-MPO)  |
| InternVL2_5-2B-MPO  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) | [internlm2_5-1_8b-chat](https://huggingface.co/internlm/internlm2_5-1_8b-chat) | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-2B-MPO)  | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-2B-MPO)  |
| InternVL2_5-4B-MPO  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |     [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)     | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-4B-MPO)  | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-4B-MPO)  |
| InternVL2_5-8B-MPO  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |   [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat)   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO)  | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-8B-MPO)  |
| InternVL2_5-26B-MPO |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |  [internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-26B-MPO) | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-26B-MPO) |
| InternVL2_5-38B-MPO |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |    [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-38B-MPO) | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-38B-MPO) |
| InternVL2_5-78B-MPO |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |    [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-78B-MPO) | [🤖️ link](https://www.modelscope.cn/models/OpenGVLab/InternVL2_5-78B-MPO) |

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
```

<br>
<br>
