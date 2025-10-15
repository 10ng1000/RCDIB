CUDA_VISIBLE_DEVICES=1 clip_benchmark eval \
    --dataset=webdatasets.txt \
    --dataset_root=/home/h3c/lyl/data/wds_{dataset_cleaned} \
    --model=ViT-B-16 \
    --pretrained/home/h3c/lyl/open_clip/scripts/logs/2025_09_17-03_13_00-model_ViT-B-16-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_20.pt \
    --output "{dataset}_{pretrained}_{model}_{language}_{task}.json" \