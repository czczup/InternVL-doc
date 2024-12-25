# Evaluation Data Preparation

> COCO images are used in VQAv2, OK-VQA, RefCOCO, POPE, and so on. Make sure you have already downloaded COCO images before evaluating on these benchmarks.

## Image Caption Benchmarks

### COCO

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/coco && cd data/coco

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip && unzip test2015.zip

# Step 3: Download and place the annotation files
mkdir -p annotations && cd annotations/
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test.json
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test_gt.json

cd ../../..
```

After preparation is complete, the directory structure is:

```
data/coco
├── annotations
│   ├── coco_karpathy_test.json
│   └── coco_karpathy_test_gt.json
├── train2014
├── val2014
└── test2015
```

### Flickr30K

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/flickr30k && cd data/flickr30k

# Step 2: Download and unzip image files
# Download images from https://bryanplummer.com/Flickr30kEntities/

# Step 3: Download and place the annotation files
# Karpathy split annotations can be downloaded from the following link:
wget https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_test_karpathy.txt
# This file is provided by the clip-benchmark repository.
# We convert this txt file to json format, download the converted file:
wget https://github.com/OpenGVLab/InternVL/releases/download/data/flickr30k_test_karpathy.json

cd ../..
```

After preparation is complete, the directory structure is:

```
data/flickr30k
├── Images
├── flickr30k_test_karpathy.txt
└── flickr30k_test_karpathy.json
```

### NoCaps

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/nocaps && cd data/nocaps

# Step 2: Download and unzip image files
# Download images from https://nocaps.org/download

# Step 3: Download and place the annotation files
# Original annotations can be downloaded from https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json

cd ../..
```

After preparation is complete, the directory structure is:

```
data/nocaps
├── images
└── nocaps_val_4500_captions.json
```

## General VQA Benchmarks

### VQAv2

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/vqav2 && cd data/vqav2

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./
ln -s ../coco/test2015 ./

# Step 3: Download questions and annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_testdev.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/vqav2
├── train2014 -> ../coco/train2014
├── val2014 -> ../coco/val2014
├── test2015 -> ../coco/test2015
├── v2_mscoco_train2014_annotations.json
├── v2_mscoco_train2014_complementary_pairs.json
├── v2_mscoco_val2014_annotations.json
├── v2_OpenEnded_mscoco_test2015_questions.json
├── v2_OpenEnded_mscoco_test-dev2015_questions.json
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── vqav2_testdev.jsonl
├── vqav2_train.jsonl
└── vqav2_val.jsonl
```

### OKVQA

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/okvqa && cd data/okvqa

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./

# Step 3: Download annotations and questions
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip && unzip mscoco_train2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip && unzip OpenEnded_mscoco_train2014_questions.json.zip
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip && unzip mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip && unzip OpenEnded_mscoco_val2014_questions.json.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/okvqa/okvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/okvqa/okvqa_val.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/okvqa
├── mscoco_train2014_annotations.json
├── mscoco_val2014_annotations.json
├── okvqa_train.jsonl
├── okvqa_val.jsonl
├── OpenEnded_mscoco_train2014_questions.json
├── OpenEnded_mscoco_val2014_questions.json
├── test2014 -> ../coco/test2014
└── val2014 -> ../coco/val2014
```

### TextVQA val

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/textvqa && cd data/textvqa

# Step 2: Download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# Step 3: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val_llava.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/textvqa
├── TextVQA_Rosetta_OCR_v0.2_test.json
├── TextVQA_Rosetta_OCR_v0.2_train.json
├── TextVQA_Rosetta_OCR_v0.2_val.json
├── textvqa_train_annotations.json
├── textvqa_train.jsonl
├── textvqa_train_questions.json
├── textvqa_val_annotations.json
├── textvqa_val.jsonl
├── textvqa_val_llava.jsonl
├── textvqa_val_questions.json
└── train_images
```

### VizWiz

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/vizwiz && cd data/vizwiz

# Step 2: Download images
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip && unzip train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip && unzip val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip && unzip test.zip

# Step 3: Download annotations
wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip && unzip Annotations.zip

# Step 4: download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/vizwiz
├── annotations
├── test
├── train
├── val
├── vizwiz_test.jsonl
├── vizwiz_train_annotations.json
├── vizwiz_train.jsonl
├── vizwiz_train_questions.json
├── vizwiz_val_annotations.json
├── vizwiz_val.jsonl
└── vizwiz_val_questions.json
```

### DocVQA

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/docvqa && cd data/docvqa

# Step 2: Download images and annotations
wget https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz --no-check-certificate # (optional)
wget https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz --no-check-certificate
wget https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz --no-check-certificate

# Step 3: Unzip files
tar -zxvf train.tar.gz
tar -zxvf val.tar.gz
tar -zxvf test.tar.gz

# Step 4: Download converted jsonl files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/docvqa
├── test
├── test.jsonl
├── train
├── train.jsonl
├── val
└── val.jsonl
```

### InfoVQA

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/infographicsvqa && cd data/infographicsvqa

# Step 2: Download images and annotations from https://rrc.cvc.uab.es/?ch=17&com=downloads
# infographicsVQA_test_v1.0.json, infographicsVQA_val_v1.0_withQT.json, infographicVQA_train_v1.0.json

# Step 3: Download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_val.jsonl -O val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_test.jsonl -O test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/infographicsvqa
├── infographicsvqa_images
├── infographicsVQA_test_v1.0.json
├── infographicsVQA_val_v1.0_withQT.json
├── infographicVQA_train_v1.0.json
├── test.jsonl
└── val.jsonl
```

### ChartQA

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/chartqa && cd data/chartqa

# Step 2: download images from
# https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view

# Step 3: download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_augmented.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_augmented.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/chartqa
 ├── ChartQA Dataset
 │    ├── test
 │    ├── train
 │    └── val
 ├── test_augmented.jsonl
 ├── test_human.jsonl
 ├── train_augmented.jsonl
 └── train_human.jsonl
```

### GQA

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/gqa && cd data/gqa

# Step 2: Download the official evaluation script
wget https://nlp.stanford.edu/data/gqa/eval.zip
unzip eval.zip

# Step 3: Download images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/testdev_balanced.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/train_balanced.jsonl
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_gqa_testdev_balanced_qwen_format.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/gqa
├── challenge_all_questions.json
├── challenge_balanced_questions.json
├── eval.py
├── images
├── llava_gqa_testdev_balanced_qwen_format.jsonl
├── readme.txt
├── submission_all_questions.json
├── test_all_questions.json
├── test_balanced.jsonl
├── test_balanced_questions.json
├── testdev_all_questions.json
├── testdev_balanced_all_questions.json
├── testdev_balanced_predictions.json
├── testdev_balanced_questions.json
├── train_all_questions
├── train_balanced.jsonl
├── train_balanced_questions.json
├── val_all_questions.json
└── val_balanced_questions.json
```

### OCRVQA

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/ocrvqa && cd data/ocrvqa

# Step 2: Download images by following instructions at 
# https://ocr-vqa.github.io/kvqa_ProjectFiles/README.txt

# Step 3: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/ocrvqa
├── images
├── ocrvqa_test.jsonl
├── ocrvqa_train.jsonl
└── ocrvqa_val.jsonl
```

### AI2D

Follow the instructions below to prepare the data：

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

After preparation is complete, the directory structure is:

```
data/ai2diagram
 ├── test_vlmevalkit.jsonl
 ├── ai2d # (optional)
 │    ├── abc_images
 │    └── images
 └── AI2D_TEST
```

### ScienceQA

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/scienceqa/images && cd data/scienceqa/images

# Step 2: download images
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip && unzip test.zip

cd ..

# Step 3: Download original questions
wget https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json

# Step 4: download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/scienceqa/scienceqa_test_img.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```
data/scienceqa
├── images
├── problems.json
└── scienceqa_test_img.jsonl
```

## Refer Expression Comprehension

### RefCOCO / RefCOCO+ / RefCOCO-g

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/refcoco && cd data/refcoco

# Step 2: Download converted files
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

After preparation is complete, the directory structure is:

```
data/refcoco
├── refcocog_test.jsonl
├── refcocog_val.jsonl
├── refcoco_testA.jsonl
├── refcoco+_testA.jsonl
├── refcoco_testB.jsonl
├── refcoco+_testB.jsonl
├── refcoco_val.jsonl
└── refcoco+_val.jsonl
```

## MultiModal Benchmarks

### MME

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/mme && cd data/mme

# Step 2: Download MME_Benchmark_release_version
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip

cd ../..
```

After preparation is complete, the directory structure is:

```
data/mme
 └── MME_Benchmark_release_version
```

### MMBench & CCBench

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/mmbench && cd data/mmbench

# Step 2: Download csv files
wget http://opencompass.openxlab.space/utils/MMBench/CCBench_legacy.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv

cd ../..
```

After preparation is complete, the directory structure is:

```

data/mmbench
 ├── CCBench_legacy.tsv
 ├── mmbench_dev_20230712.tsv
 ├── mmbench_dev_cn_20231003.tsv
 ├── mmbench_dev_en_20231003.tsv
 ├── mmbench_test_cn_20231003.tsv
 └── mmbench_test_en_20231003.tsv
```

### POPE

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/pope && cd data/pope

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/val2014 ./
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl

# Step 3: Download `coco` from POPE
mkdir -p coco && cd coco
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
cd ../../..
```

After preparation is complete, the directory structure is:

```
data/pope
├── coco
│   ├── coco_pope_adversarial.json
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json
├── llava_pope_test.jsonl
└── val2014
```

### MMMU

The evaluation script will automatically download the MMMU dataset from HuggingFace.

### MMMU-Pro

The evaluation script will automatically download the MMMU-Pro dataset from HuggingFace.

### Tiny-LVLM-eHub

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/tiny_lvlm && cd data/tiny_lvlm

# Step 2: Download the dataset
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/updated_datasets.zip
unzip updated_datasets.zip

cd ../..
```

After preparation is complete, the directory structure is:

```
data/tiny_lvlm
└── updated_datasets
```

### MMVet

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/mm-vet && cd data/mm-vet

# Step 2: Download the dataset
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/llava-mm-vet.jsonl
cd ../..
```

After preparation is complete, the directory structure is:

```
data/mm-vet
 ├── images
 └── llava-mm-vet.jsonl
```

### MMVet v2

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mm-vet-v2 && cd data/mm-vet-v2

# Step 2: Download the dataset
wget https://github.com/yuweihao/MM-Vet/releases/download/v2/mm-vet-v2.zip
unzip mm-vet-v2.zip

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mm-vet-v2
 ├── images
 └── mm-vet-v2.json
```

#### MMVP

Follow the instructions below to prepare the data:

```shell
# Step 1: Download the dataset
cd data/
git clone https://huggingface.co/datasets/MMVP/MMVP
cd ../
```

After preparation is complete, the directory structure is:

```shell
data/MMVP
 ├── MMVP Images
 ├── Questions.csv
 ├── Questions.xlsx
 └── README.md
```

### MathVista

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/MathVista && cd data/MathVista

# Step 2: Download the annotation
wget https://huggingface.co/datasets/AI4Math/MathVista/raw/main/annot_testmini.json

cd ../..
```

After preparation is complete, the directory structure is:

```
MathVista
└── annot_testmini.json
```

### SEED-Image

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/SEED && cd data/SEED

# Step 2: Download the dataset
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/SEED-Bench-image.zip
unzip SEED-Bench-image.zip
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/seed.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/SEED
 ├── SEED-Bench-image
 └── seed.jsonl
```

<br>
<br>
