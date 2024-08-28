# Deploy InternVL2 Series

## LMDeploy

[LMDeploy](https://github.com/InternLM/lmdeploy) is a toolkit for compressing, deploying, and serving LLM, developed by the MMRazor and MMDeploy teams.

```sh
pip install lmdeploy==0.5.3
```

LMDeploy abstracts the complex inference process of multi-modal Vision-Language Models (VLM) into an easy-to-use pipeline, similar to the Large Language Model (LLM) inference pipeline. Here are some deployment examples. For more detailed deployment instructions, please refer to the [lmdeploy documentation](https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/internvl.md).

### A 'Hello, world' example

`````{tabs}

````{tab} 1B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-1B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

````

````{tab} 2B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-2B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

````

````{tab} 4B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-4B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

````

````{tab} 8B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-8B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

````

````{tab} 26B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-26B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

````

````{tab} 40B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-40B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

````

````{tab} 76B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-Llama3-76B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192, tp=4))
response = pipe(('describe this image', image))
print(response.text)
```

````

`````

If `ImportError` occurs while executing this case, please install the required packages as prompted.

### Multi-images inference

When dealing with multiple images, you can put them all in one list. Keep in mind that multiple images will lead to a higher number of input tokens, and as a result, the size of the context window typically needs to be increased.

> Warning: Due to the scarcity of multi-image conversation data, the performance on multi-image tasks may be unstable, and it may require multiple attempts to achieve satisfactory results.

`````{tabs}

````{tab} 1B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2-1B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

````

````{tab} 2B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2-2B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

````

````{tab} 4B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2-4B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

````

````{tab} 8B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

````

````{tab} 26B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2-26B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

````

````{tab} 40B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2-40B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

````

````{tab} 76B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2-Llama3-76B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192, tp=4))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

````

`````

### Batch prompts inference

Conducting inference with batch prompts is quite straightforward; just place them within a list structure:

`````{tabs}

````{tab} 1B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-1B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

````

````{tab} 2B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-2B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

````

````{tab} 4B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-4B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

````

````{tab} 8B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

````

````{tab} 26B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-26B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

````

````{tab} 40B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-40B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

````

````{tab} 76B

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-Llama3-76B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192, tp=4))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

````

`````

### Multi-turn conversation

There are two ways to do the multi-turn conversations with the pipeline. One is to construct messages according to the format of OpenAI and use above introduced method, the other is to use the `pipeline.chat` interface.

`````{tabs}

````{tab} 1B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-1B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

````

````{tab} 2B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-2B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

````

````{tab} 4B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-4B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

````

````{tab} 8B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

````

````{tab} 26B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-26B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

````

````{tab} 40B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-40B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

````

````{tab} 76B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-Llama3-76B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192, tp=4))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

````

`````

### OpenAI format prompt input

Besides this structure, the pipeline also supports prompts in the OpenAI format.

Note that `lmdeploy` has introduced the `max_dynamic_patch` key in the `messages` parameter, which allows you to control the maximum number of image crops by adjusting this setting.

`````{tabs}

````{tab} 1B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

model = 'OpenGVLab/InternVL2-1B'

pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_p=0.9,
                              temperature=0.8,
                              max_new_tokens=1024)
messages  = [{
    'role': 'user',
    'content': [{
        'type:'text',
        "text':'describe this image'},
        {
            'type':'image_url',
            'image_url':{
                'max_dynamic_patch'=12,
                'url'='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
            }
        }]
}]
response = pipe(messages,gen_config=gen_config)
print(response.text)
```

````

````{tab} 2B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

model = 'OpenGVLab/InternVL2-2B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_p=0.9,
                              temperature=0.8,
                              max_new_tokens=1024)
messages  = [{
    'role': 'user',
    'content': [{
        'type:'text',
        "text':'describe this image'},
        {
            'type':'image_url',
            'image_url':{
                'max_dynamic_patch'=12,
                'url'='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
            }
        }]
}]
response = pipe(messages,gen_config=gen_config)
print(response.text)
```

````

````{tab} 4B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

model = 'OpenGVLab/InternVL2-4B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_p=0.9,
                              temperature=0.8,
                              max_new_tokens=1024)
messages  = [{
    'role': 'user',
    'content': [{
        'type:'text',
        "text':'describe this image'},
        {
            'type':'image_url',
            'image_url':{
                'max_dynamic_patch'=12,
                'url'='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
            }
        }]
}]
response = pipe(messages,gen_config=gen_config)
print(response.text)
```

````

````{tab} 8B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

model = 'OpenGVLab/InternVL2-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_p=0.9,
                              temperature=0.8,
                              max_new_tokens=1024)
messages  = [{
    'role': 'user',
    'content': [{
        'type:'text',
        "text':'describe this image'},
        {
            'type':'image_url',
            'image_url':{
                'max_dynamic_patch'=12,
                'url'='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
            }
        }]
}]
response = pipe(messages,gen_config=gen_config)
print(response.text)
```

````

````{tab} 26B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

model = 'OpenGVLab/InternVL2-26B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_p=0.9,
                              temperature=0.8,
                              max_new_tokens=1024)
messages  = [{
    'role': 'user',
    'content': [{
        'type:'text',
        "text':'describe this image'},
        {
            'type':'image_url',
            'image_url':{
                'max_dynamic_patch'=12,
                'url'='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
            }
        }]
}]
response = pipe(messages,gen_config=gen_config)
print(response.text)
```

````

````{tab} 40B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

model = 'OpenGVLab/InternVL2-40B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_p=0.9,
                              temperature=0.8,
                              max_new_tokens=1024)
messages  = [{
    'role': 'user',
    'content': [{
        'type:'text',
        "text':'describe this image'},
        {
            'type':'image_url',
            'image_url':{
                'max_dynamic_patch'=12,
                'url'='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
            }
        }]
}]
response = pipe(messages,gen_config=gen_config)
print(response.text)
```

````

````{tab} 76B

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

model = 'OpenGVLab/InternVL2-Llama3-76B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_p=0.9,
                              temperature=0.8,
                              max_new_tokens=1024)
messages  = [{
    'role': 'user',
    'content': [{
        'type:'text',
        "text':'describe this image'},
        {
            'type':'image_url',
            'image_url':{
                'max_dynamic_patch'=12,
                'url'='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
            }
        }]
}]
response = pipe(messages,gen_config=gen_config)
print(response.text)
```

````

`````

### Serving with OpenAI Compatible Server

#### Launch Service

`````{tabs}

````{tab} 1B

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-1B --backend turbomind --server-port 23333
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.
````

````{tab} 2B

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-2B --backend turbomind --server-port 23333
```

You can also load 4-bit AWQ quantized models to save memory:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-2B-AWQ --backend turbomind --server-port 23333 --model-format awq
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.
````

````{tab} 4B

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-4B --backend turbomind --server-port 23333
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.
````

````{tab} 8B

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-8B --backend turbomind --server-port 23333
```

You can also load 4-bit AWQ quantized models to save memory:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-8B-AWQ --backend turbomind --server-port 23333 --model-format awq
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.
````

````{tab} 26B

> **⚠️ Warning:** Please make sure to install Flash Attention; otherwise, using `--tp` will cause errors.

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-26B --backend turbomind --server-port 23333
```

You can also load 4-bit AWQ quantized models to save memory:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-26B-AWQ --backend turbomind --server-port 23333 --model-format awq
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.
````

````{tab} 40B

> **⚠️ Warning:** Please make sure to install Flash Attention; otherwise, using `--tp` will cause errors.

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-40B --backend turbomind --server-port 23333 --tp 2
```

You can also load 4-bit AWQ quantized models to save memory:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-40B-AWQ --backend turbomind --server-port 23333 --model-format awq
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.
````

````{tab} 76B

> **⚠️ Warning:** Please make sure to install Flash Attention; otherwise, using `--tp` will cause errors.

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 lmdeploy serve api_server OpenGVLab/InternVL2-Llama3-76B --backend turbomind --server-port 23333 --tp 4
```

You can also load 4-bit AWQ quantized models to save memory:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-Llama3-76B-AWQ --backend turbomind --server-port 23333 --model-format awq
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.
````

`````

#### Integrate with `OpenAI`

Here is an example of interaction with the endpoint `v1/chat/completions` service via the openai package. Before running it, please install the openai package by `pip install openai`.

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
                'max_dynamic_patch':"12"
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```

If you encounter any issues or need advanced usage with `lmdeploy`, we recommend reading the [lmdeploy documentation](https://lmdeploy.readthedocs.io/en/latest/serving/api_server_vl.html).

#### Memory Usage Testing

To test the memory usage with several A100 GPUs, we will consider the following variables: the number of GPUs, whether AWQ 4-bit quantization is used, and the size of `--cache-max-entry-count`. The table below shows the memory usage per GPU under different scenarios:

`````{tabs}

````{tab} 2B

| #GPUs | AWQ 4-bit | cache-max-entry-count | Memory Usage per GPU |
| :---: | :-------: | :-------------------: | :------------------: |
|   1   |    No     |          0.8          |      67140 MB        |
|   1   |    No     |          0.2          |      21284 MB        |
|   1   |    No     |          0.1          |      13700 MB        |
|   1   |    No     |          0.05         |      9860 MB         |
|   2   |    No     |          0.05         |      8612 MB         |
|   4   |    No     |          0.05         |      7916 MB         |
|       |           |                       |                      |
|   1   |    Yes    |          0.2          |      19242 MB         |
|   1   |    Yes    |          0.05         |      7850 MB         |

````

````{tab} 4B

| #GPUs | AWQ 4-bit | cache-max-entry-count | Memory Usage per GPU |
| :---: | :-------: | :-------------------: | :------------------: |
|   1   |    No     |          0.8          |      67666 MB        |
|   1   |    No     |          0.2          |      24914 MB        |
|   1   |    No     |          0.1          |      17746 MB        |
|   1   |    No     |          0.05         |      14162 MB        |
|   2   |    No     |          0.05         |      11700 MB        |
|   4   |    No     |          0.05         |      10216 MB        |

````

````{tab} 8B

| #GPUs | AWQ 4-bit | cache-max-entry-count | Memory Usage per GPU |
| :---: | :-------: | :-------------------: | :------------------: |
|   1   |    No     |          0.8          |      69708 MB        |
|   1   |    No     |          0.2          |      30624 MB        |
|   1   |    No     |          0.1          |      24108 MB        |
|   1   |    No     |          0.05         |      20832 MB        |
|   2   |    No     |          0.05         |      14570 MB        |
|   4   |    No     |          0.05         |      11378 MB        |
|       |           |                       |                      |
|   1   |    Yes    |          0.2          |      22440 MB        |
|   1   |    Yes    |          0.05         |      11528 MB        |

````

````{tab} 26B

> **⚠️ Warning:** Please make sure to install Flash Attention; otherwise, using `--tp` will cause errors.

| #GPUs | AWQ 4-bit | cache-max-entry-count | Memory Usage per GPU |
| :---: | :-------: | :-------------------: | :------------------: |
|   1   |    No     |          0.8          |       77310 MB       |
|   1   |    No     |          0.2          |       58302 MB       |
|   2   |    No     |          0.2          |       39750 MB       |
|   4   |    No     |          0.2          |       30512 MB       |
|   8   |    No     |          0.2          |       25806 MB       |
|       |           |                       |                      |
|   1   |    Yes    |          0.8          |       72104 MB       |
|   1   |    Yes    |          0.2          |       37448 MB       |
|   1   |    Yes    |          0.1          |       31656 MB       |
|   1   |    Yes    |          0.05         |       28712 MB       |

````

````{tab} 40B

> **⚠️ Warning:** Please make sure to install Flash Attention; otherwise, using `--tp` will cause errors.

| #GPUs | AWQ 4-bit | cache-max-entry-count | Memory Usage per GPU |
| :---: | :-------: | :-------------------: | :------------------: |
|   1   |    No     |          0.8          |   `Out-Of-Memory`    |
|   1   |    No     |          0.2          |       79892 MB       |
|   1   |    No     |          0.1          |     `No output`      |
|   1   |    No     |          0.05         |     `No output`      |
|   2   |    No     |          0.2          |       50156 MB       |
|   4   |    No     |          0.2          |       34990 MB       |
|   8   |    No     |          0.2          |       28052 MB       |
|       |           |                       |                      |
|   1   |    Yes    |          0.8          |       72964 MB       |
|   1   |    Yes    |          0.2          |       42628 MB       |
|   1   |    Yes    |          0.1          |       37572 MB       |
|   1   |    Yes    |          0.05         |       33636 MB       |

````

````{tab} 76B

> **⚠️ Warning:** Please make sure to install Flash Attention; otherwise, using `--tp` will cause errors.

| #GPUs | AWQ 4-bit | cache-max-entry-count | Memory Usage per GPU |
| :---: | :-------: | :-------------------: | :------------------: |
|   1   |    No     |          0.8          |   `Out-Of-Memory`    |
|   1   |    No     |          0.2          |   `Out-Of-Memory`    |
|   2   |    No     |          0.2          |       78698 MB       |
|   4   |    No     |          0.2          |       50477 MB       |
|   8   |    No     |          0.2          |       36268 MB       |
|       |           |                       |                      |
|   1   |    Yes    |          0.8          |       77138 MB       |
|   1   |    Yes    |          0.2          |       58738 MB       |
|   1   |    Yes    |          0.1          |       55698 MB       |
|   1   |    Yes    |          0.05         |       54130 MB       |

````

`````

## vLLM

Coming soon…

## Ollama

Coming soon…

<br>
<br>
