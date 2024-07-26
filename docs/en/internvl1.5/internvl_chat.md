# InternVL-Chat-V1-5

## Introduction

![image](./figure1.png)

We introduce InternVL 1.5, an open-source multimodal large language model (MLLM) to bridge the capability gap between open-source and proprietary commercial models in multimodal understanding. We introduce three simple designs:

1. Strong Vision Encoder: we explored a continuous learning strategy for the large-scale vision foundation model——InternViT-6B, boosting its visual understanding capabilities, and making it can be transferred and reused in different LLMs.

2. Dynamic High-Resolution: we divide images into tiles ranging from 1 to 40 of 448 × 448 pixels according to the aspect ratio and resolution of the input images, which supports up to 4K resolution input.

3. High-Quality Bilingual Dataset: we carefully collected a high-quality bilingual dataset that covers common scenes, document images, and annotated them with English and Chinese question-answer pairs, significantly enhancing performance in OCR- and Chinese-related tasks.

![image](./figure2.png)

As illustrated in Figure 3, InternVL 1.5 employs an architecture akin to widely-used open-source MLLMs, specifically the “ViT-MLP-LLM” configuration referenced in various existing studies. Our implementation of this architecture integrates a pre-trained InternViT-6B with a pre-trained InternLM2-20B using a randomly initialized MLP projector.

During training, we implemented a dynamic resolution strategy, dividing images into tiles of 448 × 448 pixels in sizes ranging from 1 to 12, based on the aspect ratio and resolution of the input images. During testing, this can be zero-shot scaled up to 40 tiles (i.e., 4K resolution). To enhance scalability for high resolution, we simply employed a pixel shuffle operation to reduce the number of visual tokens to one-quarter of the original. Therefore, in our model, a 448 × 448 image is represented by 256 visual tokens.

## Performance

![performance](./performance1.png)

![performance](./performance2.png)

## Quick Start

We provide an example code to run InternVL-Chat-V1-5 using `transformers`.

We also welcome you to experience the InternVL2 series models in our [online demo](https://internvl.opengvlab.com/).

> Please use transformers==4.37.2 to ensure the model works normally.

### Model Loading

#### 16-bit (bf16 / fp16)

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL-Chat-V1-5"
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
path = "OpenGVLab/InternVL-Chat-V1-5"
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
    num_layers = {'Mini-InternVL-2B-V1-5': 24, 'Mini-InternVL-4B-V1-5': 32, 'InternVL-Chat-V1-5': 48}[model_name]
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

path = "OpenGVLab/InternVL-Chat-V1-5"
device_map = split_model('InternVL-Chat-V1-5')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```

### Inference with Transformers

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
path = 'OpenGVLab/InternVL-Chat-V1-5'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
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

## Fine-tune on a Custom Dataset

### 1. Model Preparation

| model name         | type | download                                                             |  size   |
| ------------------ | ---- | -------------------------------------------------------------------- | :-----: |
| InternVL-Chat-V1-5 | MLLM | 🤗 [HF link](https://huggingface.co/NousResearch/InternVL-Chat-V1-5) | 48.0 GB |

Before starting the second fine-tuning, download the pre-trained model we provide.

```sh
cd pretrained/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL-Chat-V1-5
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-5 --local-dir InternVL-Chat-V1-5
```

The directory structure is:

```sh
pretrained
└── InternVL-Chat-V1-5
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

Fine-tune the pre-trained models using either the [script for training the full LLM](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_full.sh)
or the [script for training the LoRA adapter](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL-Chat-V1-5`.

> 💡 Fine-tuning the full LLM requires at least 8 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

> 💡 The number of GPUs and hyperparameters used here are just an example. To achieve optimal results, you may need to adjust these settings based on your available hardware and dataset size.

Commands for fine-tuning:

```sh
# Using 8 GPUs, fine-tune the full LLM, cost about 77G per GPU
GPUS=8 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_full.sh
# Using 2 GPUs, fine-tune the LoRA, cost about 79G per GPU
GPUS=2 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_lora.sh
# Using 8 GPUs, fine-tune the LoRA, cost about 60 per GPU
GPUS=8 sh shell/internvl1.5/2nd_finetune/internvl_chat_v1_5_internlm2_20b_dynamic_res_2nd_finetune_lora.sh
```

If you encounter any issues, please let me know, and I will update the training guide to enhance its usability.

## Evaluation

To evaluate the performance of the InternVL-Chat-V1-5 model across various tasks, follow the instructions for each specific dataset. Ensure that the appropriate number of GPUs is allocated as specified.

> 1⃣️ We simultaneously use InternVL and VLMEvalKit repositories for model evaluation. Specifically, the results reported for DocVQA, ChartQA, InfoVQA, TextVQA, MME, AI2D, MMBench, CCBench, MMVet, and SEED-Image were tested using the InternVL repository. OCRBench, RealWorldQA, HallBench, and MathVista were evaluated using the VLMEvalKit.

> 2⃣️ Please note that evaluating the same model using different testing toolkits like InternVL and VLMEvalKit can result in slight differences, which is normal. Updates to code versions and variations in environment and hardware can also cause minor discrepancies in results.

> 3⃣️️ Note, the dataset description is generated by GPT-4 and may contain errors.

### Evaluation using InternVL Codebase

#### Data Preparation

Please prepare the evaluation data according to the [guidance provided here](../get_started/eval_data_preparation.md).

#### MME

MME is a comprehensive benchmark designed to evaluate Multimodal Large Language Models (MLLMs) on both perception and cognition abilities across 14 different subtasks, ensuring robust and diverse testing of these models.

Please use the following command to perform the test with 1 GPU:

```bash
GPUS=1 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mme --dynamic --max-num 12
```

The expected test results are:

```
total score: 1658.3683473389356

   existence  score: 190.0
   count  score: 175.0
   position  score: 171.66666666666669
   color  score: 178.33333333333331
   posters  score: 173.8095238095238
   celebrity  score: 142.05882352941177
   scene  score: 156.5
   landmark  score: 179.5
   artwork  score: 144.0
   OCR  score: 147.5


=========== Cognition ===========
total score: 533.5714285714286

   commonsense_reasoning  score: 133.57142857142858
   numerical_calculation  score: 117.5
   text_translation  score: 185.0
   code_reasoning  score: 97.5

# 1658.3683473389356 + 533.5714285714286 = 2191.939775910364
```

#### OKVQA

OKVQA (Outside Knowledge Visual Question Answering) is a dataset designed for visual question answering tasks that require external knowledge beyond what is visible in the image, featuring over 14,000 questions to evaluate the reasoning abilities of AI models.

Please use the following command to perform the test with 8 GPU:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-okvqa-val
```

The expected test results are:

```
TODO
```

#### TextVQA

TextVQA is a dataset designed to evaluate visual question answering models by requiring them to read and reason about text present within images, containing 45,336 questions over 28,408 images from the OpenImages dataset.

The TextVQA dataset provides official OCR results, specifically Rosetta OCR tokens. During testing with InstructBLIP and LLaVA 1.5, the OCR results are input to the LLM as a prompt. If you want to input Rosetta OCR tokens, use the following command:

We do not use Rosetta OCR tokens, run this command:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-textvqa-val --dynamic --max-num 24
```

The expected test results are:

```
['pretrained/InternVL-Chat-V1-5', 'textvqa_val', 0.8061000000000043]
```

#### VizWiz

The VizWiz VQA dataset is a visual question answering dataset created to help answer visual questions posed by blind individuals. It contains over 31,000 visual questions, where users took a picture using a mobile phone and recorded a spoken question about it. Each question comes with 10 crowdsourced answers. This dataset addresses tasks such as predicting the answer to a visual question and determining whether a visual question can be answered.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-vizwiz-val
```

The expected test results are:

```
TODO
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-vizwiz-test
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission).

The expected test results are:

```
TODO
```

#### ChartQA

The ChartQA dataset is a comprehensive benchmark for question answering about charts that involves both visual and logical reasoning. It includes a mix of 9.6K human-written questions and 23.1K machine-generated questions derived from chart summaries. This dataset is designed to evaluate models that can understand and analyze charts by answering complex questions that often require multiple logical and arithmetic operations, as well as referencing visual features of the charts.

The ChartQA dataset includes two test sets: `chartqa_test_human` and `chartqa_test_augmented`. The final score for model evaluation is calculated as the average of the scores on these two test sets:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-chartqa-test
```

The expected test results are:

```
['chartqa_test_human', {'relaxed_accuracy': 0.736}]
['chartqa_test_augmented', {'relaxed_accuracy': 0.9408}]
# the average score = (73.6 + 94.08) / 2 = 83.8
```

#### DocVQA

The DocVQA dataset consists of 50,000 questions on 12,000+ document images. It is designed for visual question answering tasks where questions are answered using text within the document images. The dataset includes OCR transcriptions and ground truth answers, supporting evaluation of models that interpret and extract information from documents.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-docvqa-val
```

The expected test results are:

```
Overall ANLS: 0.9049
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-docvqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

The expected test results are:

```
Overall ANLS: 0.909
```

#### AI2D

The AI2D dataset contains over 5,000 grade school science diagrams with extensive annotations and 15,000 multiple-choice questions for research on diagram understanding and question answering.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-ai2d-test --dynamic
```

The expected test results are:

```
ai2diagram_test {'accuracy': 0.8073186528497409}
```

#### InfographicVQA

The InfographicVQA dataset is a collection of infographics accompanied by natural language questions and answers. This dataset includes a diverse range of infographics sourced from thousands of different websites, ensuring a variety of layouts and designs. It comprises 30,035 questions across 5,485 images, split into training, validation, and test sets.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-infovqa-val
```

The expected test results are:

```
Overall ANLS: 0.7235
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-infovqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

The expected test results are:

```
Overall ANLS: 0.725
```

#### GQA

The GQA dataset is a large-scale visual question answering dataset designed for real-world visual reasoning and compositional question answering. It contains over 22 million questions grounded in real images, each accompanied by detailed scene graphs that describe objects, their attributes, and relationships within the scene. The dataset includes images from the Visual Genome dataset, with questions that require various reasoning skills such as spatial understanding and multi-step inference.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-gqa-testdev
```

The expected test results are:

```
TODO
```

#### ScienceQA

The ScienceQA dataset is a large-scale benchmark for multimodal science question answering, consisting of 21,208 multiple-choice questions derived from elementary and high school science curricula. This dataset features a diverse range of topics across natural science, social science, and language science. It includes questions with image context (48.7%), text context (48.2%), and both (30.8%).

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 scienceqa
```

The expected test results are:

```
TODO
```

#### POPE

The POPE (Polling-based Object Probing Evaluation) dataset is designed to evaluate object hallucination in MLLMs. The dataset consists of 3,000 questions related to the captions of 500 images. By treating the MLLMs' answers to these questions as a binary classification task, the dataset allows researchers to measure accuracy, precision, recall, and F1 scores to determine the extent of hallucination in the models.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 pope
```

The expected test results are:

```
TODO
```

#### Tiny LVLM

The Tiny LVLM-eHub is a streamlined evaluation benchmark designed to assess the multimodal capabilities of MLLMs, including models like Bard. It focuses on six categories of multimodal abilities: visual perception, visual knowledge acquisition, visual reasoning, visual commonsense, object hallucination, and embodied intelligence.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 tiny_lvlm
```

The expected test results are:

```
TODO
```

#### MMMU

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmmu-val
```

The expected test results are:

```
{'Overall-Art and Design': {'num': 120, 'acc': 0.608}, 'Art': {'num': 30, 'acc': 0.7}, 
'Art_Theory': {'num': 30, 'acc': 0.8}, 'Design': {'num': 30, 'acc': 0.767},
'Music': {'num': 30, 'acc': 0.167}, 'Overall-Business': {'num': 150, 'acc': 0.413}, 
'Accounting': {'num': 30, 'acc': 0.467}, 'Economics': {'num': 30, 'acc': 0.4}, 
'Finance': {'num': 30, 'acc': 0.4}, 'Manage': {'num': 30, 'acc': 0.4}, 
'Marketing': {'num': 30, 'acc': 0.4}, 'Overall-Science': {'num': 150, 'acc': 0.38}, 
'Biology': {'num': 30, 'acc': 0.6}, 'Chemistry': {'num': 30, 'acc': 0.233}, 
'Geography': {'num': 30, 'acc': 0.4}, 'Math': {'num': 30, 'acc': 0.333}, 
'Physics': {'num': 30, 'acc': 0.333}, 'Overall-Health and Medicine': {'num': 150, 'acc': 0.433}, 
'Basic_Medical_Science': {'num': 30, 'acc': 0.5}, 'Clinical_Medicine': {'num': 30, 'acc': 0.5}, 
'Diagnostics_and_Laboratory_Medicine': {'num': 30, 'acc': 0.333}, 
'Pharmacy': {'num': 30, 'acc': 0.367}, 'Public_Health': {'num': 30, 'acc': 0.467}, 
'Overall-Humanities and Social Science': {'num': 120, 'acc': 0.617}, 
'History': {'num': 30, 'acc': 0.633}, 'Literature': {'num': 30, 'acc': 0.8},
'Sociology': {'num': 30, 'acc': 0.567}, 'Psychology': {'num': 30, 'acc': 0.467},
'Overall-Tech and Engineering': {'num': 210, 'acc': 0.362}, 'Agriculture': {'num': 30, 'acc': 0.567},
'Architecture_and_Engineering': {'num': 30, 'acc': 0.267}, 'Computer_Science': {'num': 30, 'acc': 0.367},
'Electronics': {'num': 30, 'acc': 0.3}, 'Energy_and_Power': {'num': 30, 'acc': 0.333}, 
'Materials': {'num': 30, 'acc': 0.467}, 'Mechanical_Engineering': {'num': 30, 'acc': 0.233}, 
'Overall': {'num': 900, 'acc': 0.452}}
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmmu-test --dynamic
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2179/overview).

The expected test results are:

```
# result of the test-en set
A_Overall (test)	0.8217488789237668
# result of the test-cn set
A_Overall (test)	0.8195067264573991
```

#### MMVet (GPT-4-0613)

The MM-Vet dataset is a comprehensive benchmark designed to evaluate the integrated capabilities of MLLMs. It encompasses six core vision-language (VL) capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and math. The dataset includes 200 images and 218 questions, each requiring one or more of these capabilities to answer. The evaluation uses an open-ended LLM-based approach, allowing assessment across various answer styles and question types.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmvet --dynamic
```

Then, submit the results to the [evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator). The expected test results are:

```
total
62.7
```

#### MMBench

The MMBench dataset is a comprehensive multi-modality benchmark designed to evaluate the fine-grained abilities of vision-language models. It contains around 3,000 multiple-choice questions covering 20 ability dimensions, structured into a hierarchical taxonomy. These dimensions include perception and reasoning abilities, further broken down into specific skills like coarse and fine-grained perception, attribute reasoning, and logic reasoning.

For the English dev / test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmbench-dev-en
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmbench-test-en

```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
mmbench-dev-en: TODO
mmbench-test-en: TODO
```

For the Chinese dev / test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmbench-dev-cn
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmbench-test-cn

```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
mmbench-dev-cn: TODO
mmbench-test-cn: TODO
```

#### CCBench

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 ccbench-dev --dynamic
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
A_Overall (dev)	0.7
```

#### SEED

CCBench is a multimodal benchmark specifically designed to evaluate models on tasks related to Chinese culture. It is part of the larger MMBench suite of benchmarks, developed by the OpenCompass Community, and aims to provide fine-grained evaluations across various capabilities of vision-language models. CCBench includes 510 questions in a multiple-choice format, focusing on cultural knowledge and understanding.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 seed
```

The expected test results are:

```
Acc@1: 0.6999444135630906
length: 17990
Accuracy for each data type:
Data type Scene Understanding: 80.37%
Data type Instance Identity: 80.45%
Data type Instance Location: 78.03%
Data type Instance Attributes: 72.39%
Data type Instances Counting: 69.19%
Data type Spatial Relation: 59.82%
Data type Instance Interaction: 77.32%
Data type Visual Reasoning: 78.85%
Data type Text Understanding: 55.81%
Data type Action Recognition: 54.08%
Data type Action Prediction: 44.82%
Data type Procedure Understanding: 40.18%
Total accuracy: 69.99%
Image accuracy: 75.99%
Video accuracy: 47.27%
```

#### MMVP

The MMVP dataset is designed to benchmark the performance of multimodal large language models (MLLMs) in visual question answering tasks. This dataset focuses on identifying "CLIP-blind pairs," which are images that appear similar to the CLIP model despite having clear visual differences. The MMVP dataset includes 300 images derived from ImageNet-1k and LAION-Aesthetics, each paired with straightforward questions to evaluate the models' visual capabilities. It highlights the challenges these systems face, often leading to incorrect responses and hallucinated explanations.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmvp
```

The expected test results are:

```
TODO
```

#### MVBench

MVBench is a comprehensive multimodal video understanding benchmark developed to evaluate the temporal comprehension capabilities of MLLMs. It includes 20 challenging video tasks that require temporal understanding and cannot be effectively solved using a single frame. The benchmark uses a novel static-to-dynamic method, transforming static tasks into dynamic ones to systematically generate video tasks that demand a wide range of temporal skills, from perception to cognition.

We evaluate our models on MVBench by extracting 16 frames from each video, and each frame was resized to a 448x448 image.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mvbench
```

The expected test results are:

```
TODO
```

### Evaluation using VLMEvalKit Codebase

#### Data Preparation

VLMEvalKit will automatically download the necessary data for evaluation, so you do not need to prepare it manually.

#### MathVista

The MathVista dataset is a comprehensive benchmark for evaluating mathematical reasoning within visual contexts. It consists of three newly created datasets—IQTest, FunctionQA, and PaperQA—designed to address logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively.

```bash
torchrun --nproc-per-node=8 run.py --data MathVista --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
TODO
```

#### HallusionBench

HallusionBench is a comprehensive benchmark designed to evaluate image-context reasoning in MLLMs, focusing on identifying issues related to language hallucination and visual illusion. The dataset consists of 346 images paired with 1,129 questions crafted by human experts. These questions are divided into two categories: Visual Dependent (VD) and Visual Supplement (VS), allowing the benchmark to assess the nuanced understanding and interpretation of visual data by MLLMs.

```bash
torchrun --nproc-per-node=8 run.py --data HallusionBench --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
2024-04-29 00:46:23,688 - Evaluation - INFO - Score:
2024-04-29 00:46:23,688 - Evaluation - INFO -           split       aAcc       fAcc       qAcc
0       Overall  66.771819  40.173410  40.879121
1            VD  63.620981  40.000000  34.296029
2            VS  71.944444  40.517241  51.123596
3     VD_figure  77.500000  65.853659  53.846154
4        VS_map  56.250000  18.181818  18.750000
5   VD_illusion  66.666667  41.935484  34.722222
6      VS_table  75.892857  46.428571  55.813953
7        VD_ocr  78.651685  58.139535  58.139535
8        VS_ocr  59.259259  38.461538  22.222222
9      VS_chart  81.538462  50.000000  72.368421
10     VD_video  51.176471  10.416667  13.043478
11      VD_math  56.481481  25.000000  27.777778
```

The final score reported in our technical report is the average: (66.771819 + 40.173410 + 40.879121) / 3 = 49.3

#### MMStar

The MMStar dataset is an advanced multimodal benchmark designed to evaluate the capabilities of MLLMs. It comprises 1,500 carefully selected samples that are balanced and purified to ensure they exhibit visual dependency and minimal data leakage. The dataset evaluates models across six core capabilities and 18 detailed axes, focusing on complex multimodal tasks that require advanced reasoning and understanding of visual content.

```bash
torchrun --nproc-per-node=8 run.py --data MMStar --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
2024-04-29 18:21:56,491 - Evaluation - INFO -   split   Overall  ...   math  science & technology
0  none  0.572667  ...  0.564                 0.408
```

#### OCRBench

OCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of MLLMs. It includes five components: Text Recognition, Scene Text-Centric Visual Question Answering (VQA), Document-Oriented VQA, Key Information Extraction (KIE), and Handwritten Mathematical Expression Recognition (HMER). The benchmark encompasses data from 29 datasets, making it one of the most thorough OCR evaluation tools available. OCRBench aims to reveal both the strengths and weaknesses of MLLMs, particularly in handling multilingual text, handwritten text, non-semantic text, and mathematical expressions. The benchmark includes 1,000 question-answer pairs, all manually verified for precision.

```bash
torchrun --nproc-per-node=8 run.py --data OCRBench --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
2024-04-29 00:28:29,681 - Evaluation - INFO - Score:
2024-04-29 00:28:29,681 - Evaluation - INFO - Text Recognition:238
2024-04-29 00:28:29,681 - Evaluation - INFO - Scene Text-centric VQA:178
2024-04-29 00:28:29,681 - Evaluation - INFO - Doc-oriented VQA:151
2024-04-29 00:28:29,681 - Evaluation - INFO - Key Information Extraction:153
2024-04-29 00:28:29,681 - Evaluation - INFO - Handwritten Mathematical Expression Recognition:4
2024-04-29 00:28:29,681 - Evaluation - INFO - Final Score:724
2024-04-29 00:28:29,681 - Evaluation - INFO - Final Score Norm:72.4
```

#### MMMU

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

```bash
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
TODO
```

#### RealWorldQA

The RealWorldQA dataset is a benchmark designed to evaluate the real-world spatial understanding capabilities of multimodal AI models. It consists of over 700 images, each accompanied by a question and a verifiable answer, focusing on various real-world scenarios, including those captured from vehicles. This dataset aims to test how well AI models comprehend physical environments and spatial relations, enhancing their ability to interpret and analyze real-world scenes.

```bash
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
2024-04-29 00:35:13,282 - Evaluation - INFO - Score:
2024-04-29 00:35:13,282 - Evaluation - INFO -   split   Overall
0  none  0.660131
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
torchrun --nproc-per-node=8 run.py --data MMVet --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
2024-04-29 18:32:38,748 - Evaluation - INFO - Score:
2024-04-29 18:32:38,748 - Evaluation - INFO -   Category  tot        acc
0      rec  187  61.818182
1      ocr  108  68.981481
2     know   84  46.428571
3      gen   80  44.875000
4     spat   75  63.600000
5     math   26  62.307692
6  Overall  218  61.513761
```

Note that because the version of GPT-4 used for scoring differs from the official server, the scores tested by VLMEvalKit will be slightly different.

#### MMMU_DEV_VAL

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

```bash
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model InternVL-Chat-V1-5 --verbose
```

The expected test results are:

```
2024-04-29 18:20:04,977 - Evaluation - INFO -         split  Overall  ...  Science  Tech & Engineering
0         dev     0.48  ...     0.36            0.428571
1  validation     0.45  ...     0.38            0.371429
```

#### VideoMME

The Video-MME dataset is a comprehensive benchmark designed to evaluate the capabilities of MLLMs in video analysis. It is the first benchmark specifically tailored for this purpose, focusing on a high-quality assessment of models' performance in processing sequential visual data.

```bash
torchrun --nproc-per-node=8 run.py --data Video-MME --model InternVL-Chat-V1-5 --verbose --nframe 16
```

The expected test results are:

```
TODO
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
