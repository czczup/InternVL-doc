# Introduction of InternVL-Chat-V1-2

We are excited to introduce [🤗 InternVL-Chat-V1-2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2). Inspired by [LLaVA-NeXT-34B](https://llava-vl.github.io/blog/2024-01-30-llava-next/), we have also adopted [Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) as the language model. Below is the pipeline.

<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/64119264f0f81eb569e0d569/GIEKCvNc1Y5iMQqLv645p.png" style="width: 70%;">
</p>

From the experimental results, we've observed that **a stronger language model (34B) can better leverage the powerful capabilities of our vision foundation model.**

For better training reproducibility, we follow the minimalist design and data efficiency similar to LLaVA-NeXT. To reduce training costs, we provide a [pre-trained MLP projector](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2/blob/main/mlp_projector/hermes_2_yi_34b.pth) and only employ around 1.2 million visual instruction tuning samples for SFT. Our model has a total of 40 billion parameters and can be trained within 1.5 days using 32 A100 GPUs. The code, data, and model have been made publicly available.

Additionally, [🤗 InternVL-Chat-V1-2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) uses the same model architecture as InternVL-Chat-V1-2, but the difference lies in the SFT dataset. InternVL-Chat-V1-2 only utilizes an SFT dataset with 1.2M samples, while our plus version employs an SFT dataset with 12M samples.

## Performance

\* Proprietary Model          † Training Set Observed

| name                        | image size | MMMU<br>(val) | MMMU<br>(test) | MathVista<br>(testmini) | MMB<br>(test) | MMB−CN<br>(test) | MMVP | MME      | ScienceQA<br>(image) | POPE | TextVQA<br>(val) | SEEDv1<br>(image) | VizWiz<br>(test) | GQA<br>(test) |
| --------------------------- | ---------- | ------------- | -------------- | ----------------------- | ------------- | ---------------- | ---- | -------- | -------------------- | ---- | ---------------- | ----------------- | ---------------- | ------------- |
| GPT-4V\*                    | unknown    | 56.8          | 55.7           | 49.9                    | 77.0          | 74.4             | 38.7 | 1409/517 | -                    | -    | 78.0             | 71.6              | -                | -             |
| Gemini Ultra\*              | unknown    | 59.4          | -              | 53.0                    | -             | -                | -    | -        | -                    | -    | 82.3             | -                 | -                | -             |
| Gemini Pro\*                | unknown    | 47.9          | -              | 45.2                    | 73.6          | 74.3             | 40.7 | 1497/437 | -                    | -    | 74.6             | 70.7              | -                | -             |
| Qwen−VL−Plus\*              | unknown    | 45.2          | 40.8           | 43.3                    | 67.0          | 70.7             | -    | 1681/502 | -                    | -    | 78.9             | 65.7              | -                | -             |
| Qwen−VL−Max\*               | unknown    | 51.4          | 46.8           | 51.0                    | 77.6          | 75.7             | -    | -        | -                    | -    | 79.5             | -                 | -                | -             |
|                             |            |               |                |                         |               |                  |      |          |                      |      |                  |                   |                  |               |
| LLaVA−NeXT−34B              | 672x672    | 51.1          | 44.7           | 46.5                    | 79.3          | 79.0             | -    | 1631/397 | 81.8                 | 87.7 | 69.5             | 75.9              | 63.8             | 67.1†         |
| InternVL−Chat<br>−V1-2      | 448x448    | 51.6          | 46.2           | 47.7                    | 82.2          | 81.2             | 56.7 | 1687/489 | 83.3                 | 88.0 | 72.5             | 75.6              | 60.0             | 64.0†         |
| InternVL−Chat<br>−V1-2−Plus | 448x448    | 50.3          | 45.6           | 59.9                    | 83.8          | 82.0             | 58.7 | 1625/553 | 98.1†                | 88.7 | 74.1†            | 76.4              | -                | 66.9†         |

- Note that we use the [official evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) to test the MMVet scores, with `GPT-4-0613` serving as the judge model. Using different versions of GPT-4 as the judge can result in significant score variations.

Here, we have conducted only a simple performance comparison. For more detailed performance information and additional evaluation metrics, please refer to our performance summary table.

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
