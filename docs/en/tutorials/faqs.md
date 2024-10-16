# FAQs

## 1. Are there performance metrics available for using InternVL2 for object detection (including single object detection capabilities)?

[https://github.com/OpenGVLab/InternVL/issues/359](https://github.com/OpenGVLab/InternVL/issues/359)

The model currently supports grounding tasks. For specific performance scores, please refer to this [link](../internvl2.0/introduction.md#grounding-benchmarks). For more general object detection and open-world detection, InternVL series models are evaluated on grounding in RefCOCO, as shown in the table below:

|             Model              | avg. | RefCOCO<br>(val) | RefCOCO<br>(testA) | RefCOCO<br>(testB) | RefCOCO+<br>(val) | RefCOCO+<br>(testA) | RefCOCO+<br>(testB) | RefCOCO‑g<br>(val) | RefCOCO‑g<br>(test) |
| :----------------------------: | :--: | :--------------: | :----------------: | :----------------: | :---------------: | :-----------------: | :-----------------: | :----------------: | :-----------------: |
| UNINEXT-H<br>(Specialist SOTA) | 88.9 |       92.6       |        94.3        |        91.5        |       85.2        |        89.6         |        79.8         |        88.7        |        89.4         |
|                                |      |                  |                    |                    |                   |                     |                     |                    |                     |
| Mini-InternVL-<br>Chat-2B-V1-5 | 75.8 |       80.7       |        86.7        |        72.9        |       72.5        |        82.3         |        60.8         |        75.6        |        74.9         |
| Mini-InternVL-<br>Chat-4B-V1-5 | 84.4 |       88.0       |        91.4        |        83.5        |       81.5        |        87.4         |        73.8         |        84.7        |        84.6         |
|       InternVL‑Chat‑V1‑5       | 88.8 |       91.4       |        93.7        |        87.1        |       87.0        |        92.3         |        80.9         |        88.5        |        89.3         |
|                                |      |                  |                    |                    |                   |                     |                     |                    |                     |
|          InternVL2‑1B          | 79.9 |       83.6       |        88.7        |        79.8        |       76.0        |        83.6         |        67.7         |        80.2        |        79.9         |
|          InternVL2‑2B          | 77.7 |       82.3       |        88.2        |        75.9        |       73.5        |        82.8         |        63.3         |        77.6        |        78.3         |
|          InternVL2‑4B          | 84.4 |       88.5       |        91.2        |        83.9        |       81.2        |        87.2         |        73.8         |        84.6        |        84.6         |
|          InternVL2‑8B          | 82.9 |       87.1       |        91.1        |        80.7        |       79.8        |        87.9         |        71.4         |        82.7        |        82.7         |
|         InternVL2‑26B          | 88.5 |       91.2       |        93.3        |        87.4        |       86.8        |        91.0         |        81.2         |        88.5        |        88.6         |
|         InternVL2‑40B          | 90.3 |       93.0       |        94.7        |        89.2        |       88.5        |        92.8         |        83.6         |        90.3        |        90.6         |
|    InternVL2-<br>Llama3‑76B    | 90.0 |       92.2       |        94.8        |        88.4        |       88.8        |        93.1         |        82.8         |        89.5        |        90.3         |

- We use the following prompt to evaluate InternVL's grounding ability: Please provide the bounding box coordinates of the region this sentence describes: `<ref>{}</ref>`

## 2. Specific format for multi-round dialogue and video in custom dataset format

[https://github.com/OpenGVLab/InternVL/issues/356](https://github.com/OpenGVLab/InternVL/issues/356)

You can prepare data according to [this document](../get_started/chat_data_format.md#multi-image-data).

Format for multiple images:

```python
{
  "id": 0,
  "image": ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"],
  "width_list": [111, 222, 333],
  "height_list": [111, 222, 333],
  "conversations": [
    {"from": "human", "value": "<image>\nuser input <image>\nuser input"},
    {"from": "gpt", "text": "assistant output"},
    {"from": "human", "value": "<image>\nuser input"},
    {"from": "gpt", "text": "assistant output"}
  ]
}
```

## 3. LORA fine-tuning issue of InternVL2

[https://github.com/OpenGVLab/InternVL/issues/350](https://github.com/OpenGVLab/InternVL/issues/350)
[https://github.com/OpenGVLab/InternVL/issues/347](https://github.com/OpenGVLab/InternVL/issues/347)

You can try updating to the latest code and then fine-tune according to the following document:

Fine-tuning InternVL 2.0: [see here](../internvl2.0/finetune.md)

Fine-tuning InternVL 1.5: [see here](../internvl1.5/finetune.md)

## 4. Excessive security hardening of the Engineering Center online demo

[https://github.com/OpenGVLab/InternVL/issues/353](https://github.com/OpenGVLab/InternVL/issues/353)

It is due to excessive security hardening, and we will continue to optimize this issue soon.

## 5. Resource configuration required for model inference, deployment, and fine-tuning

[https://github.com/OpenGVLab/InternVL/issues/79](https://github.com/OpenGVLab/InternVL/issues/79)
[https://github.com/OpenGVLab/InternVL/issues/281](https://github.com/OpenGVLab/InternVL/issues/281)
[https://github.com/OpenGVLab/InternVL/issues/283](https://github.com/OpenGVLab/InternVL/issues/283)
[https://github.com/OpenGVLab/InternVL/issues/293](https://github.com/OpenGVLab/InternVL/issues/293)
[https://github.com/OpenGVLab/InternVL/issues/295](https://github.com/OpenGVLab/InternVL/issues/295)

You can align the package versions in the dependency environment here: <https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/pyproject.toml>. Alternatively, you can try deploying this [new local demo](../get_started/local_chat_demo.md).

InternVL-1-5 is a 26B model, with model parameters consuming about 50G of memory in BF16. Considering the additional overhead during training, it requires around 100-150G. During training, you can use DeepSpeed Zero to distribute these overheads across different GPUs.


## 6. Abnormal generation results (including repetition, garbled text, etc.)

[https://github.com/OpenGVLab/InternVL/issues/289](https://github.com/OpenGVLab/InternVL/issues/289)

This issue is due to an older version of transformers, please use `transformers==4.37.2`.

## 7. Context length of each model

[https://github.com/OpenGVLab/InternVL/issues/272](https://github.com/OpenGVLab/InternVL/issues/272)

InternVL-Chat-V1-5 has a 4k context length. Mini-InternVL-Chat-2B/4B-V1-5 has an 8k context length. All models in the InternVL2 series have an 8k context length.

## 8. Perform inference faster and with less GPU memory usage using a 4-bit quantized model.

[https://github.com/OpenGVLab/InternVL/issues/250](https://github.com/OpenGVLab/InternVL/issues/250)

Using the 4-bit model quantized by AWQ is recommended, which is very fast and occupies less GPU memory than int8.

```python
from lmdeploy import pipeline
from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL-Chat-V1-5-AWQ'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
backend_config = TurbomindEngineConfig(model_format='awq')
pipe = pipeline(model, backend_config=backend_config, log_level='INFO')
response = pipe(('describe this image', image))
print(response)
```

- or service

```shell
lmdeploy serve api_server OpenGVLab/InternVL-Chat-V1-5-AWQ --backend turbomind --model-format awq
```

## 9. LMDeploy loading MiniInternVL error (due to lack of support for phi3)

[https://github.com/OpenGVLab/InternVL/issues/230](https://github.com/OpenGVLab/InternVL/issues/230)

Only LMDeploy's pytorch engine supports phi3 models, please refer to our latest README for specific usage. You can follow this document to deploy the InternVL2-4B model using lmdeploy: https://internvl.readthedocs.io/en/latest/internvl2.0/deployment.html#launch-service

## 10. How to deploy a local demo (streamlit version)

Please refer to [this document](../get_started/local_chat_demo.md#streamlit-demo).

## 11. What format should the dataset for detection and recognition be in for end-to-end OCR implementation?

https://github.com/OpenGVLab/InternVL/issues/536

The dataset format used for training OCR is as follows:

```shell
User: Please recognize the text in the image (there are dozens of similar templates like this).  
Assistant: The text in the image includes: \nXXX\nXXX\nXXX\nXXX (directly outputting the OCR results).
```

Note that bounding boxes for OCR were not utilized during training. For detailed information about the dataset format, please refer to our  [OCR Data Format Documentation](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html)

## 12. How can the input length for the model be set?

https://github.com/OpenGVLab/InternVL/issues/542

You can configure it in the model's `tokenizer_config.json`, or modify the configuration after loading the model.


<br>
<br>
