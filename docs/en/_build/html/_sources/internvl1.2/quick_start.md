# Quick Start of InternVL-Chat-V1-2

We provide an example code to run InternVL-Chat-V1-2-Plus using `transformers`.

We also welcome you to experience the InternVL2 series models in our [online demo](https://internvl.opengvlab.com/).

> Please use transformers==4.37.2 to ensure the model works normally.

## Model Preparation

| model name              | type | param | download                                                               |  size   |
| ----------------------- | ---- | ----- | ---------------------------------------------------------------------- | :-----: |
| InternVL-Chat-V1-2      | MLLM | 40.1B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)      | 75.0 GB |
| InternVL-Chat-V1-2-Plus | MLLM | 40.1B | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) | 75.0 GB |

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

## Model Loading

### 16-bit (bf16 / fp16)

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```

### BNB 8-bit Quantization

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

### BNB 4-bit Quantization

> **⚠️ Warning:** Due to significant quantization errors with BNB 4-bit quantization on InternViT-6B, the model may produce nonsensical outputs and fail to understand images. Therefore, please avoid using BNB 4-bit quantization.

### Multiple GPUs

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
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```

## Inference with Transformers

### Pure-text conversation

```python
from transformers import AutoTokenizer, AutoModel
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
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

### Single-image single-round conversation

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
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

### Single-image multi-round conversation

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
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

### Multi-image multi-round conversation, combined images

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
    use_flash_attn=True,
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

### Multi-image multi-round conversation, separate images

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
    use_flash_attn=True,
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

### Batch inference, single image per sample

```python
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from PIL import Image
import torch

path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
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

### Video multi-round conversation

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
    use_flash_attn=True,
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

### Streaming output

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
