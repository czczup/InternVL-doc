# Evaluation Data Preparation

## Image Caption Benchmarks

### COCO Karpathy test

> COCO images are used in VQAv2/OK-VQA/RefCOCO/RefCOCO+/RefCOCOg. Make sure you have already downloaded COCO images before evaluating on these benchmarks.

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/coco && cd data/coco

# download coco images
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip && unzip test2015.zip

mkdir -p annotations && cd annotations/
# download converted annotation files
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test.json
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test_gt.json

cd ../../../
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> caption-coco [--dynamic] [--max-num 6]
```

</details>

### Flickr30K Karpathy test

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/flickr30k && cd data/flickr30k

# download images from https://bryanplummer.com/Flickr30kEntities/
# karpathy split annotations can be downloaded from the following link:
# https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_test_karpathy.txt
# this file is provided by the clip-benchmark repository.
# We convert this txt file to json format, download the converted file:
wget https://github.com/OpenGVLab/InternVL/releases/download/data/flickr30k_test_karpathy.json

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> caption-flickr30k [--dynamic] [--max-num 6]
```

</details>

### NoCaps val

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/nocaps && cd data/nocaps

# download images from https://nocaps.org/download
# original annotations can be downloaded from https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> caption-nocaps [--dynamic] [--max-num 6]
```

</details>

## General VQA Benchmarks

### VQAv2 val & test-dev

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/vqav2 && cd data/vqav2

# make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./
ln -s ../coco/test2015 ./

# download questions and annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_testdev.jsonl

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# VQAv2-val
GPUS=8 sh evaluate.sh <checkpoint> vqa-vqav2-val [--dynamic] [--max-num 6]
# VQAv2-testdev
GPUS=8 sh evaluate.sh <checkpoint> vqa-vqav2-testdev [--dynamic] [--max-num 6]
```

For the testdev set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission).

</details>

### OKVQA val

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/okvqa && cd data/okvqa

# make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./

# download annotations and questions
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip && unzip mscoco_train2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip && unzip OpenEnded_mscoco_train2014_questions.json.zip
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip && unzip mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip && unzip OpenEnded_mscoco_val2014_questions.json.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/okvqa/okvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/okvqa/okvqa_val.jsonl

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> vqa-okvqa-val [--dynamic] [--max-num 6]
```

</details>

### TextVQA val

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/textvqa && cd data/textvqa

# download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val_llava.jsonl

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# without ocr tokens
GPUS=8 sh evaluate.sh <checkpoint> vqa-textvqa-val [--dynamic] [--max-num 12]
# with ocr tokens (hint: LLaVA use ocr tokens)
GPUS=8 sh evaluate.sh <checkpoint> vqa-textvqa-val-ocr [--dynamic] [--max-num 12]
```

</details>

### VizWiz val & test

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/vizwiz && cd data/vizwiz

# download images
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip && unzip train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip && unzip val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip && unzip test.zip

# download annotations
wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip && unzip Annotations.zip

# download converted files
# train
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train.jsonl
# val
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val.jsonl
# test
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_test.jsonl
cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# VizWiz val
GPUS=8 sh evaluate.sh <checkpoint> vqa-vizwiz-val [--dynamic] [--max-num 6]
# VizWiz test
GPUS=8 sh evaluate.sh <checkpoint> vqa-vizwiz-test [--dynamic] [--max-num 6]
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission).

</details>

### DocVQA val & test

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/docvqa && cd data/docvqa

# download images and annotations
wget https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz --no-check-certificate # (optional)
wget https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz --no-check-certificate
wget https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz --no-check-certificate

# unzip files
tar -zxvf train.tar.gz
tar -zxvf val.tar.gz
tar -zxvf test.tar.gz

# download converted jsonl files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/test.jsonl
cd ../..
```

After preparation is complete, the directory structure is:

```
data
 ├── docvqa
 │   ├── test
 │   ├── test.jsonl
 │   ├── train
 │   ├── train.jsonl
 │   ├── val
 │   └── val.jsonl
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# DocVQA-val
GPUS=8 sh evaluate.sh <checkpoint> vqa-docvqa-val [--dynamic] [--max-num 18]
# DocVQA-test
GPUS=8 sh evaluate.sh <checkpoint> vqa-docvqa-test [--dynamic] [--max-num 18]
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

</details>

### InfoVQA val & test

<details open>
<summary>Data Preparation</summary>

```bash
TODO
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# InfoVQA-val
GPUS=8 sh evaluate.sh <checkpoint> vqa-infovqa-val [--dynamic] [--max-num 24]
# InfoVQA-test
GPUS=8 sh evaluate.sh <checkpoint> vqa-infovqa-test [--dynamic] [--max-num 24]
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

</details>

### ChartQA test-human & test-augmented

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/chartqa && cd data/chartqa

# download images from https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_augmented.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_augmented.jsonl

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# test both ChartQA-test-human & ChartQA-test-augmented
GPUS=8 sh evaluate.sh <checkpoint> vqa-chartqa-test [--dynamic] [--max-num 6]
```

</details>

### GQA testdev

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/gqa && cd data/gqa

# download images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/testdev_balanced.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/train_balanced.jsonl
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_gqa_testdev_balanced_qwen_format.jsonl

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> vqa-gqa-testdev [--dynamic] [--max-num 6]
```

</details>

### OCRVQA val & test

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/ocrvqa && cd data/ocrvqa

# download images by following instructions at https://ocr-vqa.github.io/kvqa_ProjectFiles/README.txt

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_test.jsonl

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# OCRVQA-val
GPUS=8 sh evaluate.sh <checkpoint> vqa-ocrvqa-val [--dynamic] [--max-num 6]
# OCRVQA-test
GPUS=8 sh evaluate.sh <checkpoint> vqa-ocrvqa-test [--dynamic] [--max-num 6]
```

</details>

### AI2D test

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/ai2diagram && cd data/ai2diagram
# download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/ai2d_test_vlmevalkit.jsonl -O test_vlmevalkit.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/AI2D_TEST.zip && unzip AI2D_TEST.zip

# download images from Google drive (optional, provided by InternLM-XComposer)
# https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing
# images should be placed in `data/ai2diagram/ai2d/abc_images` and `data/ai2diagram/ai2d/images`
cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> vqa-ai2d-test [--dynamic] [--max-num 6]
```

</details>

### ScienceQA test

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/scienceqa/images && cd data/scienceqa/images

# download images
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip && unzip test.zip

cd ..

# download original questions
wget https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/scienceqa/scienceqa_test_img.jsonl

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> scienceqa [--dynamic] [--max-num 6]
```

</details>

## Refer Expression Comprehension

### RefCOCO/RefCOCO+/RefCOCO-g

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/refcoco && cd data/refcoco

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testB.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testB.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_test.jsonl

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> refcoco [--dynamic] [--max-num 6]
```

</details>

## MultiModal Benchmarks

### MME

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mme && cd data/mme

# 1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
# 2. Downloaded images to `MME_Benchmark_release_version`.

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# single GPU testing
CUDA_VISIBLE_DEVICES=0 sh evaluate.sh <checkpoint> mme [--dynamic] [--max-num 6]
```

</details>

### MMBench dev & test

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mmbench && cd data/mmbench

# download csv files of mmbench
wget http://opencompass.openxlab.space/utils/MMBench/CCBench_legacy.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
# mmbench_dev_20230712
GPUS=8 sh evaluate.sh <checkpoint> mmbench-dev-en [--dynamic] [--max-num 6]
# mmbench_dev_cn_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-dev-cn [--dynamic] [--max-num 6]
# mmbench_test_en_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-test-en [--dynamic] [--max-num 6]
# mmbench_test_cn_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-test-cn [--dynamic] [--max-num 6]
# ccbench_dev
GPUS=8 sh evaluate.sh <checkpoint> ccbench-dev [--dynamic] [--max-num 6]
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission).

</details>

### POPE

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/pope && cd data/pope

# make sure you have downloaded COCO images
ln -s ../coco/val2014 ./
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl

# download `coco` from POPE
mkdir -p coco && cd coco
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
cd ../../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> pope [--dynamic] [--max-num 6]
```

</details>

### MMMU

<details open>
<summary>Data Preparation</summary>

The evaluation code will automatically download the dataset from hugging face.

</details>

<details open>
<summary>Evaluation</summary>

```bash
# dev set
GPUS=8 sh evaluate.sh <checkpoint> mmmu-dev [--dynamic] [--max-num 6]
# val set
GPUS=8 sh evaluate.sh <checkpoint> mmmu-val [--dynamic] [--max-num 6]
# test set
GPUS=8 sh evaluate.sh <checkpoint> mmmu-test [--dynamic] [--max-num 6]
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2179/overview).

</details>

### Tiny LVLM

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/tiny_lvlm && cd data/tiny_lvlm

# download dataset from https://github.com/OpenGVLab/Multi-Modality-Arena/tree/main/tiny_lvlm_evaluation
# i.e., download `updated_datasets.tar.gz` from https://drive.google.com/file/d/1PuFC612XzOmKwzRldtBb1CFZnIjiR7we/view
tar -xzvf updated_datasets.tar.gz

cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> tiny_lvlm [--dynamic] [--max-num 6]
```

</details>

### MM-Vet

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mm-vet && cd data/mm-vet
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/llava-mm-vet.jsonl
cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> mmvet [--dynamic] [--max-num 6]
```

</details>

#### [MMVP](https://github.com/tsb0601/MMVP)

<details open>
<summary>Data Preparation</summary>

```bash
cd data
git lfs install
git clone https://huggingface.co/datasets/MMVP/MMVP
git clone https://huggingface.co/datasets/MMVP/MMVP_VLM
cd ..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> mmvp [--dynamic] [--max-num 6]
```

</details>

### MathVista

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/MathVista && cd data/MathVista
wget https://huggingface.co/datasets/AI4Math/MathVista/raw/main/annot_testmini.json
cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
export OPENAI_API_KEY='your-openai-key'
# testmini set
GPUS=8 sh evaluate.sh <checkpoint> mathvista-testmini [--dynamic] [--max-num 6]
# test set
GPUS=8 sh evaluate.sh <checkpoint> mathvista-test [--dynamic] [--max-num 6]
```

</details>

### SEED

<details open>
<summary>Data Preparation</summary>

```bash
mkdir -p data/SEED && cd data/SEED
# 1. Follow the official instructions [Data Preparation for SEED-Bench-1](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md#data-preparation-for-seed-bench-1)
#    to download the images and the videos. Put images under `./data/SEED/SEED-Bench-image`.
# 2. Extract the video frame in the middle from the downloaded videos, and put them under `./data/SEED/SEED-Bench-image`.
#    LLaVA provided the script [`extract_video_frames.py`](../internvl_chat/tools/extract_video_frames.py) modified from the official one.

wget https://huggingface.co/OpenGVLab/InternVL/raw/main/seed.jsonl
cd ../..
```

</details>

<details open>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> seed [--dynamic] [--max-num 6]
```

</details>
