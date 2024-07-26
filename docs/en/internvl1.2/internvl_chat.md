# InternVL-Chat-V1-2

## Introduction

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

## Quick Start

We provide an example code to run InternVL-Chat-V1-2-Plus using `transformers`.

We also welcome you to experience the InternVL2 series models in our [online demo](https://internvl.opengvlab.com/).

> Please use transformers==4.37.2 to ensure the model works normally.

### Model Loading

#### 16-bit (bf16 / fp16)

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
```

#### BNB 8-bit Quantization

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()
```

#### BNB 4-bit Quantization

> **⚠️ Warning:** Due to significant quantization errors with BNB 4-bit quantization on InternViT-6B, the model may produce nonsensical outputs and fail to understand images. Therefore, please avoid using BNB 4-bit quantization.

#### Multiple GPUs

The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors.

```python
import math
import torch
from transformers import AutoTokenizer, AutoModel

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {'InternVL-Chat-V1-2': 60, 'InternVL-Chat-V1-2-Plus': 60}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
device_map = split_model('InternVL-Chat-V1-2-Plus')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```

### Inference with Transformers

#### Pure-text conversation

```python
from transformers import AutoTokenizer, AutoModel
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Single-image single-round conversation

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Single-image multi-round conversation

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values = image_processor(images=image, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'Please write a poem according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Multi-image multi-round conversation, combined images

> **⚠️️ Warning:** Please note that for this model, we support multi-image chat in the interface, but the results are not very good due to the lack of training with multi-image data.

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image1 = Image.open('./examples/image1.jpg').resize((448, 448))
pixel_values1 = image_processor(images=image1, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
image2 = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values2 = image_processor(images=image2, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = '<image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Multi-image multi-round conversation, separate images

> **⚠️️ Warning:** Please note that for this model, we support multi-image chat in the interface, but the results are not very good due to the lack of training with multi-image data.

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image1 = Image.open('./examples/image1.jpg').resize((448, 448))
pixel_values1 = image_processor(images=image1, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
image2 = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values2 = image_processor(images=image2, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

generation_config = dict(max_new_tokens=1024, do_sample=False)
question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Batch inference, single image per sample

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

image_processor = CLIPImageProcessor.from_pretrained(path)
image1 = Image.open('./examples/image1.jpg').resize((448, 448))
pixel_values1 = image_processor(images=image1, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
image2 = Image.open('./examples/image2.jpg').resize((448, 448))
pixel_values2 = image_processor(images=image2, return_tensors='pt').pixel_values.to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

generation_config = dict(max_new_tokens=1024, do_sample=False)
questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}')
    print(f'Assistant: {response}')
```

#### Video multi-round conversation

> **⚠️️ Warning:** Please note that for this model, we support video chat in the interface, but the results are not very good due to the lack of training with video data.

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import torch


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    image_processor = CLIPImageProcessor.from_pretrained(path)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB').resize((448, 448))
        pixel_values = image_processor(images=img, return_tensors='pt').pixel_values
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=False)

video_path = './examples/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=8)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
question = video_prefix + 'What is the red panda doing?'
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')

question = 'Describe this video in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}')
print(f'Assistant: {response}')
```

#### Streaming output

Besides this method, you can also use the following code to get streamed output.

```python
from transformers import TextIteratorStreamer
from threading import Thread

# Initialize the streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
# Define the generation configuration
generation_config = dict(max_new_tokens=1024, do_sample=False, streamer=streamer)
# Start the model chat in a separate thread
thread = Thread(target=model.chat, kwargs=dict(
    tokenizer=tokenizer, pixel_values=pixel_values, question=question,
    history=None, return_history=False, generation_config=generation_config,
))
thread.start()

# Initialize an empty string to store the generated text
generated_text = ''
# Loop through the streamer to get the new text as it is generated
for new_text in streamer:
    if new_text == model.conv_template.sep:
        break
    generated_text += new_text
    print(new_text, end='', flush=True)  # Print each new chunk of generated text on the same line
```

## Reproduce InternVL-Chat-V1-2

Here, we provide all the necessary code, data, and models to reproduce InternVL-Chat-V1-2. Please follow the guidelines below for preparation.

### 1. Model Preparation

| model name              | type | download                                                               |  size   |
| ----------------------- | ---- | ---------------------------------------------------------------------- | :-----: |
| InternViT-6B-448px-V1-2 | ViT  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2) | 11.1 GB |
| Nous-Hermes-2-Yi-34B    | LLM  | 🤗 [HF link](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) | 65.0 GB |

If you want to replicate the training of `InternVL-Chat-V1-2`, please follow the commands below to download `InternViT-6B-448px-V1-2` and `Nous-Hermes-2-Yi-34B`.

```sh
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternViT-6B-448px-V1-2 --local-dir InternViT-6B-448px-V1-2
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/Nous-Hermes-2-Yi-34B --local-dir Nous-Hermes-2-Yi-34B
```

The directory structure is:

```sh
pretrained
├── InternViT-6B-448px-V1-2
└── Nous-Hermes-2-Yi-34B
```

### 2. Training Datasets Preparation

Inspired by LLaVA-NeXT, we adopted a data-efficient SFT strategy to train InternVL-Chat-V1-2, utilizing approximately 1.2M of visual instruction tuning samples in total, all of which are fully open-source. In a macro sense, we build upon [ShareGPT-4V](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md#prepare-images) and additionally integrate [LLaVA-ZH](https://huggingface.co/datasets/openbmb/llava_zh), [DVQA](https://github.com/kushalkafle/DVQA_dataset), [ChartQA](https://github.com/vis-nlp/ChartQA), [AI2D](https://allenai.org/data/diagrams), [DocVQA](https://www.docvqa.org/datasets), [GeoQA+](https://github.com/SCNU203/GeoQA-Plus), and [SynthDoG-EN](https://huggingface.co/datasets/naver-clova-ix/synthdog-en). Most of the data remains consistent with LLaVA-NeXT.

First, download the [annotation files](https://huggingface.co/OpenGVLab/InternVL/resolve/main/playground.zip) and place them in the `playground/opensource/` folder.

Second, download all the images we used.

- AI2D: [ai2d_images](https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing) (provided by InternLM-XComposer)
- ChartQA: [ChartQA Dataset](https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- DocVQA: [train](https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz), [val](https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz), [test](https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz)
- DVQA: [images](https://drive.google.com/file/d/1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ/view)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- LLaVA-Pretrain: [images](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing). We save all files as `.jpg`
- SAM: We only use 000000~000050.tar for now. You can quickly download 9K images from [here](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link).
- TextVQA: [trainvalimages](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- SynthDoG-EN: We only use 00000~00004 parquet files for now, with a total of 30K images. We provide the converted [images](https://huggingface.co/OpenGVLab/InternVL/resolve/main/synthdog-en-images.zip).
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- WebData: [images](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Only for academic usage.
- GeoQA+: [images](https://huggingface.co/OpenGVLab/InternVL/resolve/main/geoqa%2B_images.zip). We have converted the data format and redistributed it.

> **⚠️ Warning:** Note that in the `sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl` file, the format of the RefCOCO data is consistent with LLaVA 1.5, which is `[x1, y1, x2, y2]` with coordinates ranging from `0-1`. During the training of InternVL-Chat-V1-2, we did not apply any special processing to this format. However, for the training of InternVL-Chat-V1-2-Plus, we converted the coordinate format to `<box>[[x1, y1, x2, y2]]</box>` and adjusted the coordinate range to `0-1000`.

Then, organize the data as follows in `playground/data`:

```none
playground/
├── opensource
│   ├── ai2d_train_12k.jsonl
│   ├── chartqa_train_18k.jsonl
│   ├── docvqa_train_10k.jsonl
│   ├── dvqa_train_200k.jsonl
│   ├── geoqa+.jsonl
│   ├── llava_instruct_150k_zh.jsonl
│   ├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
│   ├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl
│   └── synthdog_en.jsonl
├── data
│   ├── ai2d
│   │   ├── abc_images
│   │   └── images
│   ├── chartqa
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── coco
│   │   └── train2017
│   ├── docvqa
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── dvqa
│   │   └── images
│   ├── gqa
│   │   └── images
│   ├── llava
│   │   └── llava_pretrain
│   │       └── images
│   ├── ocr_vqa
│   │   └── images
│   ├── sam
│   │   └── images
│   ├── share_textvqa
│   │   └── images
│   ├── synthdog-en
│   │   └── images
│   ├── textvqa
│   │   └── train_images
│   ├── vg
│   │   ├── VG_100K
│   │   └── VG_100K_2
│   ├── web-celebrity
│   │   └── images
│   ├── web-landmark
│   │   └── images
│   ├── wikiart
│   │   └── images
│   ├── geoqa+
│   │   └── images
```

### 3. Start Training

We provide slurm scripts for multi-node multi-GPU training. You can use either 32 or 64 GPUs to train this model. If you use 64 GPUs, training will take approximately 18 hours.

- If you encounter an OOM error, you can decrease the `PER_DEVICE_BATCH_SIZE`, for example, set `PER_DEVICE_BATCH_SIZE=4`.

```sh
# using 32 GPUs
PARTITION='your partition' GPUS=32 PER_DEVICE_BATCH_SIZE=8 sh shell/internvl1.2/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune.sh
# using 64 GPUs
PARTITION='your partition' GPUS=64 PER_DEVICE_BATCH_SIZE=8 sh shell/internvl1.2/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune.sh
```

The hyperparameters used for fine-tuning are listed in the following table. And, you can view the training logs in tensorboard at [here](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2/tensorboard).

| Hyperparameter         | Trainable Param | Global Batch Size | Learning rate | Epoch | Max length | Weight decay |
| ---------------------- | --------------- | ----------------- | ------------- | ----- | ---------- | ------------ |
| InternVL-Chat-<br>V1-2 | 40B             | 512               | 1e-5          | 1     | 2048       | 0.05         |

## Fine-tune on a Custom Dataset

### 1. Model Preparation

| model name              | type | download                                                               |  size   |
| ----------------------- | ---- | ---------------------------------------------------------------------- | :-----: |
| InternVL-Chat-V1-2      | MLLM | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)      | 75.0 GB |
| InternVL-Chat-V1-2-Plus | MLLM | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) | 75.0 GB |

Before starting the second fine-tuning, download the pre-trained model we provide. Two versions are available: [InternVL-Chat-V1-2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2) and [InternVL-Chat-V1-2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus). We recommend downloading the Plus version.

Use the following commands to download the desired model:

```shell
cd pretrained/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL-Chat-V1-2
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-2 --local-dir InternVL-Chat-V1-2
# Download OpenGVLab/InternVL-Chat-V1-2-Plus
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-2-Plus --local-dir InternVL-Chat-V1-2-Plus
```

The directory structure is:

```sh
pretrained
├── InternVL-Chat-V1-2
└── InternVL-Chat-V1-2-Plus
```

### 2. Prepare Your Customized Training Data

After downloading the pre-trained model, prepare your customized SFT (Supervised Fine-Tuning) data. Create a JSON file in `internvl_chat/shell/data/` similar to [this example](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/data/internvl_1_2_finetune.json).

The format for the JSON file should be:

```json
{
  "your-custom-dataset-1": {
    "root": "path/to/the/image/",
    "annotation": "path/to/the/jsonl/annotation",
    "data_augment": false,
    "repeat_time": 1,
    "length": "number of your data"
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

My suggestion is to add new domain-specific data on top of the [general data from our open-sourced InternVL 1.2](../internvl1.2/internvl_chat.md#training-datasets-preparation). This will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

### 3. Start 2nd Fine-tuning

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.2/2nd_finetune/internvl_chat_v1_2_hermes2_yi34b_448_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.2/2nd_finetune/internvl_chat_v1_2_hermes2_yi34b_448_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL-Chat-V1-2-Plus`.

> 💡 Fine-tuning the full LLM requires at least 16 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 16 GPUs with SLURM system, fine-tune the full LLM, cost about 80G per GPU
PARTITION='your partition' GPUS=16 sh shell/internvl1.2/2nd_finetune/internvl_chat_v1_2_hermes2_yi34b_448_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, without SLURM system, cost about 63G per GPU
GPUS=2 sh shell/internvl1.2/2nd_finetune/internvl_chat_v1_2_hermes2_yi34b_448_res_2nd_finetune_lora.sh
```

If you encounter any issues, please let me know, and I will update the training guide to enhance its usability.

## Evaluation

To evaluate the performance of the InternVL-Chat-V1-2-Plus model across various tasks, follow the instructions for each specific dataset. Ensure that the appropriate number of GPUs is allocated as specified.

> 1⃣️ We simultaneously use InternVL and VLMEvalKit repositories for model evaluation. For certain datasets like MMVet and LLaVA-Bench, different GPT-4 versions used as judges cause significant result discrepancies between two codebases.

> 2⃣️ Please note that evaluating the same model using different testing toolkits like InternVL and VLMEvalKit can result in slight differences, which is normal. Updates to code versions and variations in environment and hardware can also cause minor discrepancies in results.

> 3⃣️️ Note, the dataset description is generated by GPT-4 and may contain errors.

### Evaluation using InternVL Codebase

#### Data Preparation

Please prepare the evaluation data according to the [guidance provided here](../get_started/eval_data_preparation.md).

#### MME

MME is a comprehensive benchmark designed to evaluate Multimodal Large Language Models (MLLMs) on both perception and cognition abilities across 14 different subtasks, ensuring robust and diverse testing of these models.

Please use the following command to perform the test with 1 GPU:

```bash
GPUS=1 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mme
```

The expected test results are:

```
TODO
```

#### OKVQA

OKVQA (Outside Knowledge Visual Question Answering) is a dataset designed for visual question answering tasks that require external knowledge beyond what is visible in the image, featuring over 14,000 questions to evaluate the reasoning abilities of AI models.

Please use the following command to perform the test with 8 GPU:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-okvqa-val
```

The expected test results are:

```
TODO
```

#### TextVQA

TextVQA is a dataset designed to evaluate visual question answering models by requiring them to read and reason about text present within images, containing 45,336 questions over 28,408 images from the OpenImages dataset.

The TextVQA dataset provides official OCR results, specifically Rosetta OCR tokens. During testing with InstructBLIP and LLaVA 1.5, the OCR results are input to the LLM as a prompt. If you want to input Rosetta OCR tokens, use the following command:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-textvqa-val-ocr
```

The expected test results are:

```
TODO
```

If you do not want to input Rosetta OCR tokens, use this command:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-textvqa-val
```

The expected test results are:

```
TODO
```

#### VizWiz

The VizWiz VQA dataset is a visual question answering dataset created to help answer visual questions posed by blind individuals. It contains over 31,000 visual questions, where users took a picture using a mobile phone and recorded a spoken question about it. Each question comes with 10 crowdsourced answers. This dataset addresses tasks such as predicting the answer to a visual question and determining whether a visual question can be answered.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-vizwiz-val
```

The expected test results are:

```
TODO
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-vizwiz-test
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/overview).

The expected test results are:

```
TODO
```

#### ChartQA

The ChartQA dataset is a comprehensive benchmark for question answering about charts that involves both visual and logical reasoning. It includes a mix of 9.6K human-written questions and 23.1K machine-generated questions derived from chart summaries. This dataset is designed to evaluate models that can understand and analyze charts by answering complex questions that often require multiple logical and arithmetic operations, as well as referencing visual features of the charts.

The ChartQA dataset includes two test sets: `chartqa_test_human` and `chartqa_test_augmented`. The final score for model evaluation is calculated as the average of the scores on these two test sets:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-chartqa-test
```

The expected test results are:

```
['chartqa_test_human', {'relaxed_accuracy': }]
['chartqa_test_augmented', {'relaxed_accuracy': }]
average score = ( + ) / 2 = 
```

#### DocVQA

The DocVQA dataset consists of 50,000 questions on 12,000+ document images. It is designed for visual question answering tasks where questions are answered using text within the document images. The dataset includes OCR transcriptions and ground truth answers, supporting evaluation of models that interpret and extract information from documents.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-docvqa-val
```

The expected test results are:

```
Overall ANLS: 
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-docvqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

The expected test results are:

```
TODO
```

#### AI2D

The AI2D dataset contains over 5,000 grade school science diagrams with extensive annotations and 15,000 multiple-choice questions for research on diagram understanding and question answering.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-ai2d-test
```

The expected test results are:

```
TODO
```

#### InfographicVQA

The InfographicVQA dataset is a collection of infographics accompanied by natural language questions and answers. This dataset includes a diverse range of infographics sourced from thousands of different websites, ensuring a variety of layouts and designs. It comprises 30,035 questions across 5,485 images, split into training, validation, and test sets.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-infovqa-val
```

The expected test results are:

```
TODO
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-infovqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

The expected test results are:

```
TODO
```

#### GQA

The GQA dataset is a large-scale visual question answering dataset designed for real-world visual reasoning and compositional question answering. It contains over 22 million questions grounded in real images, each accompanied by detailed scene graphs that describe objects, their attributes, and relationships within the scene. The dataset includes images from the Visual Genome dataset, with questions that require various reasoning skills such as spatial understanding and multi-step inference.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-gqa-testdev
```

The expected test results are:

```
TODO
```

#### ScienceQA

The ScienceQA dataset is a large-scale benchmark for multimodal science question answering, consisting of 21,208 multiple-choice questions derived from elementary and high school science curricula. This dataset features a diverse range of topics across natural science, social science, and language science. It includes questions with image context (48.7%), text context (48.2%), and both (30.8%).

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus scienceqa
```

The expected test results are:

```
TODO
```

#### POPE

The POPE (Polling-based Object Probing Evaluation) dataset is designed to evaluate object hallucination in MLLMs. The dataset consists of 3,000 questions related to the captions of 500 images. By treating the MLLMs' answers to these questions as a binary classification task, the dataset allows researchers to measure accuracy, precision, recall, and F1 scores to determine the extent of hallucination in the models.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus pope
```

The expected test results are:

```
TODO
```

#### Tiny LVLM

The Tiny LVLM-eHub is a streamlined evaluation benchmark designed to assess the multimodal capabilities of MLLMs, including models like Bard. It focuses on six categories of multimodal abilities: visual perception, visual knowledge acquisition, visual reasoning, visual commonsense, object hallucination, and embodied intelligence.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus tiny_lvlm
```

The expected test results are:

```
TODO
```

#### MMMU

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmmu-val
```

The expected test results are:

```
TODO
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmmu-test
```

Then submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2179/overview). The expected test results are:

```
TODO
```

#### MMVet (GPT-4-0613)

> **⚠️ Warning:** Here, we use `GPT-4-0613` as the judge model, while in VLMEvalKit, `GPT-4-Turbo` is used as the judge model. Using different versions of GPT-4 can result in significant score variations. Therefore, testing the same model with the two codebases can lead to notable score differences.

The MM-Vet dataset is a comprehensive benchmark designed to evaluate the integrated capabilities of MLLMs. It encompasses six core vision-language (VL) capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and math. The dataset includes 200 images and 218 questions, each requiring one or more of these capabilities to answer. The evaluation uses an open-ended LLM-based approach, allowing assessment across various answer styles and question types.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmvet
```

Then, submit the results to the [evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator). The expected test results are:

```
TODO
```

#### MMBench

The MMBench dataset is a comprehensive multi-modality benchmark designed to evaluate the fine-grained abilities of vision-language models. It contains around 3,000 multiple-choice questions covering 20 ability dimensions, structured into a hierarchical taxonomy. These dimensions include perception and reasoning abilities, further broken down into specific skills like coarse and fine-grained perception, attribute reasoning, and logic reasoning.

For the English dev / test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmbench-dev-en
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmbench-test-en

```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
mmbench-dev-en: TODO
mmbench-test-en: TODO
```

For the Chinese dev / test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmbench-dev-cn
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmbench-test-cn

```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
mmbench-dev-cn: TODO
mmbench-test-cn: TODO
```

#### CCBench

CCBench, a multi-modal benchmark in the domain of Chinese Culture, is designed to evaluate the performance of MLLMs on tasks specifically related to Chinese cultural content.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus ccbench-dev
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
ccbench-dev: TODO
```

#### SEED

CCBench is a multimodal benchmark specifically designed to evaluate models on tasks related to Chinese culture. It is part of the larger MMBench suite of benchmarks, developed by the OpenCompass Community, and aims to provide fine-grained evaluations across various capabilities of vision-language models. CCBench includes 510 questions in a multiple-choice format, focusing on cultural knowledge and understanding.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus seed
```

The expected test results are:

```
TODO
```

#### MMVP

The MMVP dataset is designed to benchmark the performance of multimodal large language models (MLLMs) in visual question answering tasks. This dataset focuses on identifying "CLIP-blind pairs," which are images that appear similar to the CLIP model despite having clear visual differences. The MMVP dataset includes 300 images derived from ImageNet-1k and LAION-Aesthetics, each paired with straightforward questions to evaluate the models' visual capabilities. It highlights the challenges these systems face, often leading to incorrect responses and hallucinated explanations.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmvp
```

The expected test results are:

```
TODO
```

#### LLaVA-Bench (GPT-4-0613)

> **⚠️ Warning:** Here, we use `GPT-4-0613` as the judge model, while in VLMEvalKit, `GPT-4-Turbo` is used as the judge model. Using different versions of GPT-4 can result in significant score variations. Therefore, testing the same model with the two codebases can lead to notable score differences.

The LLaVA-Bench-in-the-Wild dataset is designed to evaluate the capabilities of MLLMs in handling more complex and diverse visual tasks. It includes a set of 24 images with 60 associated questions, covering a range of indoor and outdoor scenes, memes, paintings, and sketches. Each image is paired with detailed, manually curated descriptions and questions that test the model's generalizability to novel domains.

```bash
export OPENAI_API_KEY='your openai key'
GPUS=1 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus llava-bench
```

The expected test results are:

```

```

#### MVBench

MVBench is a comprehensive multimodal video understanding benchmark developed to evaluate the temporal comprehension capabilities of MLLMs. It includes 20 challenging video tasks that require temporal understanding and cannot be effectively solved using a single frame. The benchmark uses a novel static-to-dynamic method, transforming static tasks into dynamic ones to systematically generate video tasks that demand a wide range of temporal skills, from perception to cognition.

We evaluate our models on MVBench by extracting 16 frames from each video, and each frame was resized to a 448x448 image.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mvbench
```

The expected test results are:

```
TODO
```

### Evaluation using VLMEvalKit Codebase

#### Data Preparation

VLMEvalKit will automatically download the data for evaluation, so you do not need to prepare it manually.

#### MathVista

The MathVista dataset is a comprehensive benchmark for evaluating mathematical reasoning within visual contexts. It consists of three newly created datasets—IQTest, FunctionQA, and PaperQA—designed to address logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively.

```bash
torchrun --nproc-per-node=8 run.py --data MathVista_MINI --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
TODO
```

#### HallusionBench

HallusionBench is a comprehensive benchmark designed to evaluate image-context reasoning in MLLMs, focusing on identifying issues related to language hallucination and visual illusion. The dataset consists of 346 images paired with 1,129 questions crafted by human experts. These questions are divided into two categories: Visual Dependent (VD) and Visual Supplement (VS), allowing the benchmark to assess the nuanced understanding and interpretation of visual data by MLLMs.

```bash
torchrun --nproc-per-node=8 run.py --data HallusionBench --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
TODO
```

#### MMStar

The MMStar dataset is an advanced multimodal benchmark designed to evaluate the capabilities of MLLMs. It comprises 1,500 carefully selected samples that are balanced and purified to ensure they exhibit visual dependency and minimal data leakage. The dataset evaluates models across six core capabilities and 18 detailed axes, focusing on complex multimodal tasks that require advanced reasoning and understanding of visual content.

```bash
torchrun --nproc-per-node=8 run.py --data MMStar --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
TODO
```

#### OCRBench

OCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of MLLMs. It includes five components: Text Recognition, Scene Text-Centric Visual Question Answering (VQA), Document-Oriented VQA, Key Information Extraction (KIE), and Handwritten Mathematical Expression Recognition (HMER). The benchmark encompasses data from 29 datasets, making it one of the most thorough OCR evaluation tools available. OCRBench aims to reveal both the strengths and weaknesses of MLLMs, particularly in handling multilingual text, handwritten text, non-semantic text, and mathematical expressions. The benchmark includes 1,000 question-answer pairs, all manually verified for precision.

```bash
torchrun --nproc-per-node=8 run.py --data OCRBench --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
TODO
```

#### MMMU

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

```bash
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
TODO
```

#### RealWorldQA

The RealWorldQA dataset is a benchmark designed to evaluate the real-world spatial understanding capabilities of multimodal AI models. It consists of over 700 images, each accompanied by a question and a verifiable answer, focusing on various real-world scenarios, including those captured from vehicles. This dataset aims to test how well AI models comprehend physical environments and spatial relations, enhancing their ability to interpret and analyze real-world scenes.

```bash
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
TODO
```

#### LLaVA-Bench (GPT-4-Turbo)

The LLaVA-Bench-in-the-Wild dataset is designed to evaluate the capabilities of MLLMs in handling more complex and diverse visual tasks. It includes a set of 24 images with 60 associated questions, covering a range of indoor and outdoor scenes, memes, paintings, and sketches. Each image is paired with detailed, manually curated descriptions and questions that test the model's generalizability to novel domains.

```bash
torchrun --nproc-per-node=8 run.py --data LLaVABench --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
TODO
```

#### MMVet (GPT-4-Turbo)

The MM-Vet dataset is a comprehensive benchmark designed to evaluate the integrated capabilities of MLLMs. It encompasses six core vision-language (VL) capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and math. The dataset includes 200 images and 218 questions, each requiring one or more of these capabilities to answer. The evaluation uses an open-ended LLM-based approach, allowing assessment across various answer styles and question types.

```bash
torchrun --nproc-per-node=8 run.py --data MMVet --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```

```

#### MMMU_DEV_VAL

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

```bash
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```

```

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
