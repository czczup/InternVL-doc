# Evaluation of InternVL-Chat-V1-2

To evaluate the performance of the InternVL-Chat-V1-2-Plus model across various tasks, follow the instructions for each specific dataset. Ensure that the appropriate number of GPUs is allocated as specified.

> 1⃣️ We simultaneously use InternVL and VLMEvalKit repositories for model evaluation. For certain datasets like MMVet and LLaVA-Bench, different GPT-4 versions used as judges cause significant result discrepancies between two codebases.

> 2⃣️ Please note that evaluating the same model using different testing toolkits like InternVL and VLMEvalKit can result in slight differences, which is normal. Updates to code versions and variations in environment and hardware can also cause minor discrepancies in results.

> 3⃣️️ Note, the dataset description is generated by GPT-4 and may contain errors.

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

## Evaluation using InternVL Codebase

### Data Preparation

Please prepare the evaluation data according to the [guidance provided here](../get_started/eval_data_preparation.md).

### MME

MME is a comprehensive benchmark designed to evaluate Multimodal Large Language Models (MLLMs) on both perception and cognition abilities across 14 different subtasks, ensuring robust and diverse testing of these models.

Please use the following command to perform the test with 1 GPU:

```bash
GPUS=1 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mme
```

The expected test results are:

```
=========== Perception ===========
total score: 1614.0419167667067

         existence  score: 190.0
         count  score: 155.0
         position  score: 178.33333333333331
         color  score: 180.0
         posters  score: 184.69387755102042
         celebrity  score: 176.76470588235293
         scene  score: 157.0
         landmark  score: 164.0
         artwork  score: 118.25
         OCR  score: 110.0


=========== Cognition ===========
total score: 558.2142857142858

         commonsense_reasoning  score: 155.71428571428572
         numerical_calculation  score: 132.5
         text_translation  score: 185.0
         code_reasoning  score: 85.0
```

### OKVQA

OKVQA (Outside Knowledge Visual Question Answering) is a dataset designed for visual question answering tasks that require external knowledge beyond what is visible in the image, featuring over 14,000 questions to evaluate the reasoning abilities of AI models.

Please use the following command to perform the test with 8 GPU:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-okvqa-val
```

The expected test results are:

```
okvqa_val 0.6763864447086718
```

### TextVQA

TextVQA is a dataset designed to evaluate visual question answering models by requiring them to read and reason about text present within images, containing 45,336 questions over 28,408 images from the OpenImages dataset.

The TextVQA dataset provides official OCR results, specifically Rosetta OCR tokens. During testing with InstructBLIP and LLaVA 1.5, the OCR results are input to the LLM as a prompt. If you want to input Rosetta OCR tokens, use the following command:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-textvqa-val-ocr
```

The expected test results are:

```
textvqa_val_ocr 0.7410400000000032
```

If you do not want to input Rosetta OCR tokens, use this command:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-textvqa-val
```

The expected test results are:

```
textvqa_val 0.7118800000000035
```

### VizWiz

The VizWiz VQA dataset is a visual question answering dataset created to help answer visual questions posed by blind individuals. It contains over 31,000 visual questions, where users took a picture using a mobile phone and recorded a spoken question about it. Each question comes with 10 crowdsourced answers. This dataset addresses tasks such as predicting the answer to a visual question and determining whether a visual question can be answered.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-vizwiz-val
```

The expected test results are:

```
vizwiz_val 0.6134950914563562
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-vizwiz-test
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2185/overview).

The expected test results are:

```
vizwiz_test 0.595
```

### ChartQA

The ChartQA dataset is a comprehensive benchmark for question answering about charts that involves both visual and logical reasoning. It includes a mix of 9.6K human-written questions and 23.1K machine-generated questions derived from chart summaries. This dataset is designed to evaluate models that can understand and analyze charts by answering complex questions that often require multiple logical and arithmetic operations, as well as referencing visual features of the charts.

The ChartQA dataset includes two test sets: `chartqa_test_human` and `chartqa_test_augmented`. The final score for model evaluation is calculated as the average of the scores on these two test sets:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-chartqa-test
```

The expected test results are:

```
['chartqa_test_human', {'relaxed_accuracy': 0.5772}]
['chartqa_test_augmented', {'relaxed_accuracy': 0.8796}]
average score = (57.72 + 87.96) / 2 = 72.8
```

### DocVQA

The DocVQA dataset consists of 50,000 questions on 12,000+ document images. It is designed for visual question answering tasks where questions are answered using text within the document images. The dataset includes OCR transcriptions and ground truth answers, supporting evaluation of models that interpret and extract information from documents.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-docvqa-val
```

The expected test results are:

```
Overall ANLS: 0.5689
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-docvqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

The expected test results are:

```
Overall ANLS: 0.5680
```

### AI2D

The AI2D dataset contains over 5,000 grade school science diagrams with extensive annotations and 15,000 multiple-choice questions for research on diagram understanding and question answering.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-ai2d-test
```

The expected test results are:

```
ai2diagram_test {'accuracy': 0.7888031088082902}
```

### InfographicVQA

The InfographicVQA dataset is a collection of infographics accompanied by natural language questions and answers. This dataset includes a diverse range of infographics sourced from thousands of different websites, ensuring a variety of layouts and designs. It comprises 30,035 questions across 5,485 images, split into training, validation, and test sets.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-infovqa-val
```

The expected test results are:

```
Overall ANLS: 0.4093
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-infovqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

The expected test results are:

```
Overall ANLS: 0.406
```

### GQA

The GQA dataset is a large-scale visual question answering dataset designed for real-world visual reasoning and compositional question answering. It contains over 22 million questions grounded in real images, each accompanied by detailed scene graphs that describe objects, their attributes, and relationships within the scene. The dataset includes images from the Visual Genome dataset, with questions that require various reasoning skills such as spatial understanding and multi-step inference.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus vqa-gqa-testdev
```

The expected test results are:

```
Accuracy: 66.91%
```

### ScienceQA

The ScienceQA dataset is a large-scale benchmark for multimodal science question answering, consisting of 21,208 multiple-choice questions derived from elementary and high school science curricula. This dataset features a diverse range of topics across natural science, social science, and language science. It includes questions with image context (48.7%), text context (48.2%), and both (30.8%).

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus scienceqa
```

The expected test results are:

```
Acc@1: 0.9806727813584531
```

### POPE

The POPE (Polling-based Object Probing Evaluation) dataset is designed to evaluate object hallucination in MLLMs. The dataset consists of 3,000 questions related to the captions of 500 images. By treating the MLLMs' answers to these questions as a binary classification task, the dataset allows researchers to measure accuracy, precision, recall, and F1 scores to determine the extent of hallucination in the models.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus pope
```

The expected test results are:

```
Category: random, # samples: 2910
TP      FP      TN      FN
1230    18      1392    270
Accuracy: 0.9010309278350516
Precision: 0.9855769230769231
Recall: 0.82
F1 score: 0.8951965065502183
Yes ratio: 0.4288659793814433
0.895, 0.901, 0.986, 0.820, 0.429
====================================
Category: popular, # samples: 3000
TP      FP      TN      FN
1230    42      1458    270
Accuracy: 0.896
Precision: 0.9669811320754716
Recall: 0.82
F1 score: 0.8874458874458875
Yes ratio: 0.424
0.887, 0.896, 0.967, 0.820, 0.424
====================================
Category: adversarial, # samples: 3000
TP      FP      TN      FN
1230    77      1423    270
Accuracy: 0.8843333333333333
Precision: 0.9410864575363428
Recall: 0.82
F1 score: 0.8763804773779836
Yes ratio: 0.43566666666666665
0.876, 0.884, 0.941, 0.820, 0.436
====================================

(89.5 + 88.7 + 87.6) / 3 = 88.6
```

### Tiny LVLM

The Tiny LVLM-eHub is a streamlined evaluation benchmark designed to assess the multimodal capabilities of MLLMs, including models like Bard. It focuses on six categories of multimodal abilities: visual perception, visual knowledge acquisition, visual reasoning, visual commonsense, object hallucination, and embodied intelligence.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus tiny_lvlm
```

The expected test results are:

```
Visual_Knowledge_Acquisition: 0.75
Object_Hallucination: 0.89
Visual_Commonsense: 0.638
Visual_Perception: 0.5625
Visual_Reasoning: 0.6909090909090909
Overall: 3.53909090909091
```

### MMMU

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

For the validation set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmmu-val
```

The expected test results are:

```
{'Overall-Art and Design': {'num': 120, 'acc': 0.542}, 'Art': {'num': 30, 'acc': 0.667}, 'Art_Theory': {'num': 30, 'acc': 0.633}, 'Design': {'num': 30, 'acc': 0.7}, 'Music': {'num': 30, 'acc': 0.167}, 'Overall-Business': {'num': 150, 'acc': 0.46}, 'Accounting': {'num': 30, 'acc': 0.567}, 'Economics': {'num': 30, 'acc': 0.467}, 'Finance': {'num': 30, 'acc': 0.367}, 'Manage': {'num': 30, 'acc': 0.367}, 'Marketing': {'num': 30, 'acc': 0.533}, 'Overall-Science': {'num': 150, 'acc': 0.38}, 'Biology': {'num': 30, 'acc': 0.4}, 'Chemistry': {'num': 30, 'acc': 0.2}, 'Geography': {'num': 30, 'acc': 0.6}, 'Math': {'num': 30, 'acc': 0.4}, 'Physics': {'num': 30, 'acc': 0.3}, 'Overall-Health and Medicine': {'num': 150, 'acc': 0.573}, 'Basic_Medical_Science': {'num': 30, 'acc': 0.5}, 'Clinical_Medicine': {'num': 30, 'acc': 0.633}, 'Diagnostics_and_Laboratory_Medicine': {'num': 30, 'acc': 0.467}, 'Pharmacy': {'num': 30, 'acc': 0.533}, 'Public_Health': {'num': 30, 'acc': 0.733}, 'Overall-Humanities and Social Science': {'num': 120, 'acc': 0.708}, 'History': {'num': 30, 'acc': 0.7}, 'Literature': {'num': 30, 'acc': 0.833}, 'Sociology': {'num': 30, 'acc': 0.7}, 'Psychology': {'num': 30, 'acc': 0.6}, 'Overall-Tech and Engineering': {'num': 210, 'acc': 0.419}, 'Agriculture': {'num': 30, 'acc': 0.433}, 'Architecture_and_Engineering': {'num': 30, 'acc': 0.4}, 'Computer_Science': {'num': 30, 'acc': 0.467}, 'Electronics': {'num': 30, 'acc': 0.233}, 'Energy_and_Power': {'num': 30, 'acc': 0.567}, 'Materials': {'num': 30, 'acc': 0.367}, 'Mechanical_Engineering': {'num': 30, 'acc': 0.467}, 'Overall': {'num': 900, 'acc': 0.5}}
```

For the test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmmu-test
```

Then submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2179/overview). The expected test results are:

```
All subject resultes
{'Overall-Art & Design': {'num': 1163, 'acc': 0.595}, 'Art': {'num': 231, 'acc': 0.658}, 'Art_Theory': {'num': 429, 'acc': 0.648}, 'Design': {'num': 169, 'acc': 0.822}, 'Music': {'num': 334, 'acc': 0.368}, 'Overall-Business': {'num': 1428, 'acc': 0.405}, 'Accounting': {'num': 380, 'acc': 0.453}, 'Economics': {'num': 267, 'acc': 0.449}, 'Finance': {'num': 355, 'acc': 0.324}, 'Manage': {'num': 245, 'acc': 0.347}, 'Marketing': {'num': 181, 'acc': 0.475}, 'Overall-Science': {'num': 2426, 'acc': 0.38}, 'Biology': {'num': 345, 'acc': 0.412}, 'Chemistry': {'num': 603, 'acc': 0.31}, 'Geography': {'num': 565, 'acc': 0.444}, 'Math': {'num': 505, 'acc': 0.392}, 'Physics': {'num': 408, 'acc': 0.353}, 'Overall-Health & Medicine': {'num': 1752, 'acc': 0.501}, 'Basic_Medical_Science': {'num': 326, 'acc': 0.586}, 'Clinical_Medicine': {'num': 325, 'acc': 0.542}, 'Diagnostics_and_Laboratory_Medicine': {'num': 162, 'acc': 0.475}, 'Pharmacy': {'num': 430, 'acc': 0.493}, 'Public_Health': {'num': 509, 'acc': 0.434}, 'Overall-Humanities & Social Science': {'num': 947, 'acc': 0.713}, 'History': {'num': 278, 'acc': 0.752}, 'Literature': {'num': 112, 'acc': 0.866}, 'Sociology': {'num': 252, 'acc': 0.714}, 'Psychology': {'num': 305, 'acc': 0.62}, 'Overall-Tech & Engineering': {'num': 2784, 'acc': 0.377}, 'Agriculture': {'num': 287, 'acc': 0.355}, 'Architecture_and_Engineering': {'num': 551, 'acc': 0.312}, 'Computer_Science': {'num': 371, 'acc': 0.412}, 'Electronics': {'num': 256, 'acc': 0.305}, 'Energy_and_Power': {'num': 432, 'acc': 0.41}, 'Materials': {'num': 458, 'acc': 0.356}, 'Mechanical_Engineering': {'num': 429, 'acc': 0.476}, 'Overall': {'num': 10500, 'acc': 0.456}}

Leaderboard
[{'test_split': {'Art & Design': 0.595, 'Business': 0.405, 'Science': 0.38, 'Health & Medicine': 0.501, 'Humanities & Social Science': 0.713, 'Tech & Engineering': 0.377, 'Overall': 0.456}}]
```

### MMVet (GPT-4-0613)

> **⚠️ Warning:** Here, we use `GPT-4-0613` as the judge model, while in VLMEvalKit, `GPT-4-Turbo` is used as the judge model. Using different versions of GPT-4 can result in significant score variations. Therefore, testing the same model with the two codebases can lead to notable score differences.

The MM-Vet dataset is a comprehensive benchmark designed to evaluate the integrated capabilities of MLLMs. It encompasses six core vision-language (VL) capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and math. The dataset includes 200 images and 218 questions, each requiring one or more of these capabilities to answer. The evaluation uses an open-ended LLM-based approach, allowing assessment across various answer styles and question types.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmvet
```

Then, submit the results to the [evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator). The expected test results are:

```
runs: [47.9]
```

### MMBench

The MMBench dataset is a comprehensive multi-modality benchmark designed to evaluate the fine-grained abilities of vision-language models. It contains around 3,000 multiple-choice questions covering 20 ability dimensions, structured into a hierarchical taxonomy. These dimensions include perception and reasoning abilities, further broken down into specific skills like coarse and fine-grained perception, attribute reasoning, and logic reasoning.

For the English dev / test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmbench-dev-en
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmbench-test-en

```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
mmbench-dev-en: 83.4
mmbench-test-en: 83.8
```

For the Chinese dev / test set, run:

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmbench-dev-cn
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmbench-test-cn

```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
mmbench-dev-cn: 81.6
mmbench-test-cn: 82.0
```

### CCBench

CCBench, a multi-modal benchmark in the domain of Chinese Culture, is designed to evaluate the performance of MLLMs on tasks specifically related to Chinese cultural content.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus ccbench-dev
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are:

```
ccbench-dev: 55.9
```

### SEED

CCBench is a multimodal benchmark specifically designed to evaluate models on tasks related to Chinese culture. It is part of the larger MMBench suite of benchmarks, developed by the OpenCompass Community, and aims to provide fine-grained evaluations across various capabilities of vision-language models. CCBench includes 510 questions in a multiple-choice format, focusing on cultural knowledge and understanding.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus seed
```

The expected test results are:

```
Data type Scene Understanding: 80.24%
Data type Instance Identity: 79.90%
Data type Instance Location: 77.95%
Data type Instance Attributes: 71.37%
Data type Instances Counting: 72.25%
Data type Spatial Relation: 63.01%
Data type Instance Interaction: 77.32%
Data type Visual Reasoning: 79.46%
Data type Text Understanding: 47.67%
Data type Action Recognition: 49.11%
Data type Action Prediction: 41.80%
Data type Procedure Understanding: 52.59%
Total accuracy: 70.43%
Image accuracy: 76.44%
Video accuracy: 47.67%
```

### MMVP

The MMVP dataset is designed to benchmark the performance of multimodal large language models (MLLMs) in visual question answering tasks. This dataset focuses on identifying "CLIP-blind pairs," which are images that appear similar to the CLIP model despite having clear visual differences. The MMVP dataset includes 300 images derived from ImageNet-1k and LAION-Aesthetics, each paired with straightforward questions to evaluate the models' visual capabilities. It highlights the challenges these systems face, often leading to incorrect responses and hallucinated explanations.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mmvp
```

The expected test results are:

```
Evaluating MMVP ...
Results saved to results/MMVP_240727004726.jsonl
The accuracy is 0.5866666666666667
```

### LLaVA-Bench (GPT-4-0613)

> **⚠️ Warning:** Here, we use `GPT-4-0613` as the judge model, while in VLMEvalKit, `GPT-4-Turbo` is used as the judge model. Using different versions of GPT-4 can result in significant score variations. Therefore, testing the same model with the two codebases can lead to notable score differences.

The LLaVA-Bench-in-the-Wild dataset is designed to evaluate the capabilities of MLLMs in handling more complex and diverse visual tasks. It includes a set of 24 images with 60 associated questions, covering a range of indoor and outdoor scenes, memes, paintings, and sketches. Each image is paired with detailed, manually curated descriptions and questions that test the model's generalizability to novel domains.

```bash
export OPENAI_API_KEY='your openai key'
GPUS=1 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus llava-bench
```

The expected test results are:

```
all *85.0* 87.0 73.9
llava_bench_complex [8.75, 7.429] 84.9
llava_bench_complex 84.9 87.5 74.3
llava_bench_conv [8.824, 7.706] 87.3
llava_bench_conv 87.3 88.2 77.1
llava_bench_detail [8.467, 6.967] 82.3
llava_bench_detail 82.3 84.7 69.7
```

### MathVista

The MathVista dataset is a comprehensive benchmark for evaluating mathematical reasoning within visual contexts. It consists of three newly created datasets—IQTest, FunctionQA, and PaperQA—designed to address logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively.

```bash
export OPENAI_API_KEY='your-openai-key'
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mathvista-testmini
```

The expected test results are:

```
Correct: 597, Total: 1000, Accuracy: 59.7%
1000
Number of test problems: 1000

Type: [question_type]
[free_form]: 52.39% (241/460)
[multi_choice]: 65.93% (356/540)

Type: [answer_type]
[float]: 0.00% (0/40)
[integer]: 57.42% (240/418)
[text]: 65.93% (356/540)
[list]: 50.00% (1/2)

Type: [language]
[english]: 58.33% (546/936)
[chinese]: 82.26% (51/62)
[persian]: 0.00% (0/2)
```

### RefCOCO Series

RefCOCO, RefCOCO+, and RefCOCOg are datasets used for tasks involving referring expression comprehension, segmentation, and generation. These datasets are built upon the MSCOCO dataset, and they are essential for evaluating models in natural language processing and computer vision.

```bash
GPUS=8 sh evalulate.sh pretrained/InternVL-Chat-V1-2-Plus refcoco
```

The expected test results are:

```
RefCOCO val, 90.2
RefCOCO testA, 93.4
RefCOCO testB, 85.5
RefCOCO+ val, 85.3
RefCOCO+ testA, 90.4
RefCOCO+ testB, 79.7
RefCOCO‑g val, 88.5
RefCOCO‑g test, 88.8
```

### MVBench

MVBench is a comprehensive multimodal video understanding benchmark developed to evaluate the temporal comprehension capabilities of MLLMs. It includes 20 challenging video tasks that require temporal understanding and cannot be effectively solved using a single frame. The benchmark uses a novel static-to-dynamic method, transforming static tasks into dynamic ones to systematically generate video tasks that demand a wide range of temporal skills, from perception to cognition.

We evaluate our models on MVBench by extracting 16 frames from each video, and each frame was resized to a 448x448 image.

```bash
GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-2-Plus mvbench --load-in-8bit
```

The expected test results are:

```
{'Action Sequence': 54.50000000000001, 'Action Prediction': 54.0, 'Action Antonym': 48.5, 
'Fine-grained Action': 39.0, 'Unexpected Action': 79.0, 'Object Existence': 48.5, 
'Object Interaction': 62.5, 'Object Shuffle': 39.5, 'Moving Direction': 35.5, 
'Action Localization': 32.5, 'Scene Transition': 88.0, 'Action Count': 42.0, 'Moving Count': 38.0,
'Moving Attribute': 60.5, 'State Change': 47.0, 'Fine-grained Pose': 53.5, 'Character Order': 68.5, 
'Egocentric Navigation': 28.999999999999996, 'Episodic Reasoning': 63.5,  'Counterfactual Inference': 36.0, 'Avg': 50.975}
```

## Evaluation using VLMEvalKit Codebase

### Data Preparation

VLMEvalKit will automatically download the data for evaluation, so you do not need to prepare it manually.

### MathVista

The MathVista dataset is a comprehensive benchmark for evaluating mathematical reasoning within visual contexts. It consists of three newly created datasets—IQTest, FunctionQA, and PaperQA—designed to address logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively.

```bash
torchrun --nproc-per-node=8 run.py --data MathVista_MINI --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
--  ---------------------------  ----  ---  ---  -------  -------
 0  Overall                      1000  665  575  66.5     57.5
 1  scientific reasoning          122   94   68  77.0492  55.7377
 2  textbook question answering   158  112   80  70.8861  50.6329
 3  numeric commonsense           144   66   66  45.8333  45.8333
 4  arithmetic reasoning          353  210  224  59.4901  63.4561
 5  visual question answering     179   99   96  55.3073  53.6313
 6  geometry reasoning            239  151  129  63.1799  53.9749
 7  algebraic reasoning           281  177  131  62.9893  46.6192
 8  geometry problem solving      208  136  110  65.3846  52.8846
 9  math word problem             186  145  152  77.957   81.7204
10  logical reasoning              37   22    5  59.4595  13.5135
11  figure question answering     269  173  137  64.3123  50.9294
12  statistical reasoning         301  203  187  67.4419  62.1262
--  ---------------------------  ----  ---  ---  -------  -------
```

### HallusionBench

HallusionBench is a comprehensive benchmark designed to evaluate image-context reasoning in MLLMs, focusing on identifying issues related to language hallucination and visual illusion. The dataset consists of 346 images paired with 1,129 questions crafted by human experts. These questions are divided into two categories: Visual Dependent (VD) and Visual Supplement (VS), allowing the benchmark to assess the nuanced understanding and interpretation of visual data by MLLMs.

```bash
torchrun --nproc-per-node=8 run.py --data HallusionBench --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
--  -----------  -------  -------  -------
 0  Overall      65.51    41.6185  37.1429
 1  VD           62.9442  41.7391  32.13
 2  VS           69.7222  41.3793  44.9438
 3  VD_ocr       79.7753  65.1163  60.4651
 4  VS_chart     70       37.5     56.5789
 5  VD_figure    73.75    60.9756  46.1538
 6  VS_map       59.375   40.9091  18.75
 7  VD_illusion  61.1111  38.7097  27.7778
 8  VS_table     80.3571  53.5714  55.814
 9  VD_math      54.6296  22.2222  29.6296
10  VS_ocr       59.2593  34.6154  25.9259
11  VD_video     55.8824  22.9167  13.0435
--  -----------  -------  -------  -------

result = (65.51 + 41.6185 + 37.1429) / 3 = 48.1
```

### MMStar

The MMStar dataset is an advanced multimodal benchmark designed to evaluate the capabilities of MLLMs. It comprises 1,500 carefully selected samples that are balanced and purified to ensure they exhibit visual dependency and minimal data leakage. The dataset evaluates models across six core capabilities and 18 detailed axes, focusing on complex multimodal tasks that require advanced reasoning and understanding of visual content.

```bash
torchrun --nproc-per-node=8 run.py --data MMStar --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
-----------------------  -----
split                    none
Overall                  0.604
coarse perception        0.676
fine-grained perception  0.528
instance reasoning       0.676
logical reasoning        0.616
math                     0.712
science & technology     0.416
-----------------------  -----
```

### OCRBench

OCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of MLLMs. It includes five components: Text Recognition, Scene Text-Centric Visual Question Answering (VQA), Document-Oriented VQA, Key Information Extraction (KIE), and Handwritten Mathematical Expression Recognition (HMER). The benchmark encompasses data from 29 datasets, making it one of the most thorough OCR evaluation tools available. OCRBench aims to reveal both the strengths and weaknesses of MLLMs, particularly in handling multilingual text, handwritten text, non-semantic text, and mathematical expressions. The benchmark includes 1,000 question-answer pairs, all manually verified for precision.

```bash
torchrun --nproc-per-node=8 run.py --data OCRBench --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
{
    "Text Recognition": 255,
    "Scene Text-centric VQA": 164,
    "Doc-oriented VQA": 92,
    "Key Information Extraction": 82,
    "Handwritten Mathematical Expression Recognition": 5,
    "Final Score": 598,
    "Final Score Norm": 59.8
}
```

### MMMU

The MMMU dataset is a comprehensive benchmark designed to evaluate multimodal models on college-level tasks that require domain-specific knowledge and reasoning. It includes 11,500 questions sourced from college exams, quizzes, and textbooks, spanning six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions cover 30 subjects and feature 30 types of images, such as charts, diagrams, maps, tables, and more.

```bash
torchrun --nproc-per-node=8 run.py --data MMMU_DEV_VAL --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
-----------------------------------  -------------------  -------------------
split                                validation           dev
Overall                              0.5188888888888888   0.52
Accounting                           0.5333333333333333   0.6
Agriculture                          0.43333333333333335  0.2
Architecture_and_Engineering         0.3333333333333333   0.2
Art                                  0.6666666666666666   1.0
Art_Theory                           0.6666666666666666   0.8
Basic_Medical_Science                0.5333333333333333   1.0
Biology                              0.3333333333333333   0.8
Chemistry                            0.4                  0.0
Clinical_Medicine                    0.6333333333333333   0.6
Computer_Science                     0.5666666666666667   0.6
Design                               0.7333333333333333   0.6
Diagnostics_and_Laboratory_Medicine  0.4666666666666667   0.4
Economics                            0.5                  0.2
Electronics                          0.36666666666666664  0.4
Energy_and_Power                     0.5666666666666667   0.6
Finance                              0.5                  0.2
Geography                            0.5666666666666667   0.2
History                              0.7                  1.0
Literature                           0.8333333333333334   0.6
Manage                               0.5666666666666667   0.6
Marketing                            0.5                  0.4
Materials                            0.36666666666666664  0.4
Math                                 0.36666666666666664  0.6
Mechanical_Engineering               0.4666666666666667   0.6
Music                                0.16666666666666666  0.4
Pharmacy                             0.5333333333333333   0.6
Physics                              0.3                  0.2
Psychology                           0.6                  0.8
Public_Health                        0.7                  0.4
Sociology                            0.6666666666666666   0.6
Art & Design                         0.5583333333333333   0.7
Business                             0.52                 0.4
Health & Medicine                    0.5733333333333334   0.6
Humanities & Social Science          0.7                  0.75
Science                              0.3933333333333333   0.36
Tech & Engineering                   0.44285714285714284  0.42857142857142855
-----------------------------------  -------------------  -------------------
```

### RealWorldQA

The RealWorldQA dataset is a benchmark designed to evaluate the real-world spatial understanding capabilities of multimodal AI models. It consists of over 700 images, each accompanied by a question and a verifiable answer, focusing on various real-world scenarios, including those captured from vehicles. This dataset aims to test how well AI models comprehend physical environments and spatial relations, enhancing their ability to interpret and analyze real-world scenes.

```bash
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
-------  ------------------
split    none
Overall  0.6775882352941176
-------  ------------------
```

### MMVet (GPT-4-Turbo)

The MM-Vet dataset is a comprehensive benchmark designed to evaluate the integrated capabilities of MLLMs. It encompasses six core vision-language (VL) capabilities: recognition, knowledge, optical character recognition (OCR), spatial awareness, language generation, and math. The dataset includes 200 images and 218 questions, each requiring one or more of these capabilities to answer. The evaluation uses an open-ended LLM-based approach, allowing assessment across various answer styles and question types.

```bash
torchrun --nproc-per-node=8 run.py --data MMVet --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
-  -------  ---  -------
0  rec      187  50.8556
1  ocr      108  50.7407
2  know      84  35.7143
3  gen       80  34.5
4  spat      75  47.6
5  math      26  18.8462
6  Overall  218  47.156
-  -------  ---  -------
```

Note that because the version of GPT-4 used for scoring differs from the official server, the scores tested by VLMEvalKit will be slightly different.

### LLaVA-Bench (GPT-4-Turbo)

The LLaVA-Bench-in-the-Wild dataset is designed to evaluate the capabilities of MLLMs in handling more complex and diverse visual tasks. It includes a set of 24 images with 60 associated questions, covering a range of indoor and outdoor scenes, memes, paintings, and sketches. Each image is paired with detailed, manually curated descriptions and questions that test the model's generalizability to novel domains.

```bash
torchrun --nproc-per-node=8 run.py --data LLaVABench --model InternVL-Chat-V1-2-Plus --verbose
```

The expected test results are:

```
-  -------  ----  ----  ----
0  overall *76.4* 59.3  77.7
1  complex  75.2  59.6  79.3
2  conv     86.3  74.1  85.9
3  detail   64.3  42    65.3
-  -------  ----  ----  ----
```

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
