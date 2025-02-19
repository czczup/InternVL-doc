# Quick Start of InternVL 2.5 Series

> Please use transformers>=4.37.2 to ensure the model works normally.

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

Download the above model weights according to your need and place them in the `pretrained/` folder.

```sh
pip install -U huggingface_hub

cd pretrained/
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

## Model Loading

### 16-bit (bf16 / fp16)

`````{tabs}

````{tab} 1B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-1B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```
````

````{tab} 2B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-2B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```
````

````{tab} 4B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-4B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```
````

````{tab} 8B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```
````

````{tab} 26B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-26B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```
````

````{tab} 38B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-38B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```
````

````{tab} 78B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-78B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```
````

`````

### BNB 8-bit Quantization

`````{tabs}

````{tab} 1B


```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-1B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 2B


```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-2B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 4B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-4B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 8B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 26B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-26B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 38B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-38B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 78B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-78B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

`````

### BNB 4-bit Quantization

`````{tabs}

````{tab} 1B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-1B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 2B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-2B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 4B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-4B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 8B

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

````

````{tab} 26B

> **⚠️ Warning:** Due to significant quantization errors with BNB 4-bit quantization on InternViT-6B, the model may produce nonsensical outputs and fail to understand images. Therefore, please avoid using BNB 4-bit quantization.

````

````{tab} 38B

> **⚠️ Warning:** Due to significant quantization errors with BNB 4-bit quantization on InternViT-6B, the model may produce nonsensical outputs and fail to understand images. Therefore, please avoid using BNB 4-bit quantization.

````

````{tab} 78B

> **⚠️ Warning:** Due to significant quantization errors with BNB 4-bit quantization on InternViT-6B, the model may produce nonsensical outputs and fail to understand images. Therefore, please avoid using BNB 4-bit quantization.

````

`````

### Multiple GPUs

The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors.

```python
import math
import torch
from transformers import AutoTokenizer, AutoModel

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
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
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
```

`````{tabs}

````{tab} 1B
```python
path = "OpenGVLab/InternVL2_5-1B"
device_map = split_model('InternVL2_5-1B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```
````

````{tab} 2B
```python
path = "OpenGVLab/InternVL2_5-2B"
device_map = split_model('InternVL2_5-2B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```
````

````{tab} 4B
```python
path = "OpenGVLab/InternVL2_5-4B"
device_map = split_model('InternVL2_5-4B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```
````

````{tab} 8B
```python
path = "OpenGVLab/InternVL2_5-8B"
device_map = split_model('InternVL2_5-8B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```
````

````{tab} 26B
```python
path = "OpenGVLab/InternVL2_5-26B"
device_map = split_model('InternVL2_5-26B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```
````

````{tab} 38B
```python
path = "OpenGVLab/InternVL2_5-38B"
device_map = split_model('InternVL2_5-38B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```
````

````{tab} 78B
```python
path = "OpenGVLab/InternVL2_5-78B"
device_map = split_model('InternVL2_5-78B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```
````

`````

## Inference with Transformers

```python
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OpenGVLab/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=False)

# pure-text conversation (纯文本对话)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# single-image single-round conversation (单图单轮对话)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# single-image multi-round conversation (单图多轮对话)
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Please write a poem according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

question = '<image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# batch inference, single image per sample (单图批处理)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')

# video multi-round conversation (视频多轮对话)
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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

video_path = './examples/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
question = video_prefix + 'What is the red panda doing?'
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Describe this video in detail. Don\'t repeat.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
```

### Streaming Output

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
