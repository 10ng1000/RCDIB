CUDA_VISIBLE_DEVICES=1 clip_benchmark eval \
    --dataset=webdatasets.txt \
    --dataset_root=/home/h3c/lyl/data/wds_{dataset_cleaned} \
    --model=ViT-B-16 \
   --pretrained/home/h3c/lyl/open_clip/scripts/logs/2025_09_17-03_13_00-model_ViT-B-16-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_20.pt \
    --output "{dataset}_{pretrained}_{model}_{language}_{task}.json" \
    #RN50--pretrained=/home/h3c/lyl/open_clip/scripts/rn50/logs/2025_09_17-10_50_44-model_RN50-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_8.pt
    #RN101--pretrained=/home/h3c/lyl/open_clip/scripts/logs/2025_09_17-00_07_56-model_RN101-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_5.pt
        # --pretrained=/home/h3c/lyl/open_clip/scripts/logs/2025_08_15-20_41_09-model_ViT-B-16-lr_1e-06-b_92-j_4-p_amp/checkpoints/epoch_30.pt
    # --pretrained=/home/h3c/lyl/open_clip/src/open_clip_train/logs/2025_07_26-11_42_23-model_ViT-B-16-lr_1e-06-b_92-j_4-p_amp/checkpoints/epoch_30.pt \ ours