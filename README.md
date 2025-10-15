This repository contains the source code of RCDIB: [ROBUST CLIP DISTILLATION VIA INFORMATION BOTTLENECK.]

# Install
```
conda create -n RCDIB python=3.8
conda activate RCDIB
pip install -r requirements.txt
```

# Dataset preparation
## Training datasets

Training datasets are available  on [MSCOCO](https://huggingface.co/datasets/clip-benchmark/wds_mscoco_captions) and [Flickr30K](https://huggingface.co/datasets/clip-benchmark/wds_flickr30k). We use the `train` split to finetune the CLIP models.

## Evaluation Datasets

To evaluate our models on downstream classification tasks, we evaluate the models on [MSCOCO](https://huggingface.co/datasets/clip-benchmark/wds_mscoco_captions) and [Flickr30K](https://huggingface.co/datasets/clip-benchmark/wds_flickr30k) test split.

For zero-shot evaluation, we use [imagenet1k](https://huggingface.co/datasets/clip-benchmark/wds_imagenet1k), [imagenet-r](https://huggingface.co/datasets/clip-benchmark/wds_imagenet-r), [imagenetv2](https://huggingface.co/datasets/clip-benchmark/wds_imagenetv2) and [imagenet_sketch](https://huggingface.co/datasets/clip-benchmark/wds_imagenet_sketch).

# Distill the CLIP models

Firstly, run `scripts/run_pretrain.sh` to pretrain the IBM models on the teacher CLIP model. Then use `scripts/run_distill.sh` to distill the CLIP teacher to the student.

| Network | MSCOCO | Flickr30K | ImageNet | Download |
| :----: | :----: | :----: | :----: |:----:|
|  ViT-B/16 |76.48| 96.20 |67.01|[model](https://github.com/10ng1000/RCDIB/releases/download/RCDIBv1.0/ViT-B-16_MSCOCO_Flickr30k.pt)|
|  ViT-B/32 |73.72| 93.60 |61.32|[model](https://github.com/10ng1000/RCDIB/releases/download/RCDIBv1.0/ViT-B-32_MSCOCO_Flickr30k.pt)|
|  RN101 |72.98| 94.00 |60.69|[model](https://github.com/10ng1000/RCDIB/releases/download/RCDIBv1.0/RN101_MSCOCO_Flickr30k.pt)|
|  RN50 |70.40| 92.80 |58.21|[model](https://github.com/10ng1000/RCDIB/releases/download/RCDIBv1.0/RN50_MSCOCO_Flickr30k.pt)|

# Evaluation
To evaluation the models on classification tasks, use `scripts/test_coco.sh` and `scripts/test_flickr.sh`. For zero-shot classfication, use `scripts/zero-shot-eval/test_imagenet.sh`.










# Acknowledgement
Our codebase is built over [open_clip](https://github.com/mlfoundations/open_clip). Many thanks for the contributors!