# Introduction of InternVL-Chat-V1-1

We released [🤗 InternVL-Chat-V1-1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1), featuring a structure similar to LLaVA, including a ViT, an MLP projector, and an LLM.
As shown in the figure below, we connected our InternViT-6B to LLaMA2-13B through a simple MLP projector. Note that the LLaMA2-13B used here is not the original model but an internal chat version obtained by incrementally pre-training and fine-tuning the LLaMA2-13B base model for Chinese language tasks. Overall, our model has a total of 19 billion parameters.

<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/64119264f0f81eb569e0d569/HD29tU-g0An9FpQn1yK8X.png" style="width: 75%;">
</p>

In this version, we explored increasing the resolution to 448 × 448, enhancing OCR capabilities, and improving support for Chinese conversations. Since the 448 × 448 input image generates 1024 visual tokens after passing through the ViT, leading to a significant computational burden, we use a pixel shuffle (unshuffle) operation to reduce the 1024 tokens to 256 tokens.

For more detailed information about this model, please read our [blog](https://internvl.github.io/blog/2024-01-24-InternVL-1.1/).

## Performance

|             model              |  LLaVA-1.5   | InternVL-Chat-V1-0 | InternVL-Chat-V1-0 | InternVL-Chat-V1-1 |
| :----------------------------: | :----------: | :----------------: | :----------------: | :----------------: |
|           resolution           |     336      |        336         |        448         |        448         |
|         vision encoder         | CLIP-L-336px | InternViT-6B-224px | InternViT-6B-448px | InternViT-6B-448px |
|         language model         |  Vicuna-13B  |     Vicuna-13B     |     Vicuna-13B     |     LLaMA2-13B     |
|                                |              |                    |                    |                    |
|    VQAv2<sub>testdev</sub>     |     80.0     |        80.2        |        82.0        |        80.9        |
|     GQA<sub>testdev</sub>      |     63.3     |        63.9        |        64.1        |        62.5        |
|     VizWiz<sub>test</sub>      |     53.6     |        54.6        |        60.1        |        57.3        |
|       SQA<sub>test</sub>       |     71.6     |        70.1        |        71.6        |        90.1        |
| TextVQA<sub>val, w/o OCR</sub> |      -       |         -          |         -          |        64.2        |
| TextVQA<sub>val, w/ OCR</sub>  |     61.3     |        58.7        |        64.8        |        68.6        |
|              POPE              |     85.9     |        87.1        |        87.2        |        87.1        |
|    MME<sub>perception</sub>    |    1531.3    |       1546.9       |       1579.0       |       1659.8       |
|     MMB-EN<sub>test</sub>      |     67.7     |        66.5        |        68.2        |        75.4        |
|     MMB-CN<sub>test</sub>      |     63.6     |        61.9        |        64.0        |        70.3        |
|   MMVet<sub>GPT-4-0613</sub>   |     35.4     |        33.7        |        36.7        |        46.7        |

- Note that we use the [official evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) to test the MMVet scores, with `GPT-4-0613` serving as the judge model. Using different versions of GPT-4 as the judge can result in significant score variations.

Here, we have conducted only a simple performance comparison. For more detailed performance information and additional evaluation metrics, please refer to our [performance summary table](<>).

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

<br>
<br>
