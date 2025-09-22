torchrun --nproc_per_node 2 -m open_clip_train.main \
   --report-to tensorboard \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
 	 --train-data="/home/h3c/lyl/data/flickr30k/train/{0..4}.tar::/home/h3c/lyl/data/wds_mscoco_captions/train/{0..13}.tar" \
    --val-data="/home/h3c/lyl/data/wds_mscoco_captions/test/{0..4}.tar" \
    --warmup 800 \
    --batch-size 64 \
    --lr=1e-6 \
    --wd=0.1 \
    --epochs=30 \
    --workers=4 \
    --seed=42 \
    --model=ViT-B-16 \
    --train-num-samples 111783 \
    --distill-model=ViT-L-14 \
    --distill-pretrained /home/h3c/lyl/open_clip/scripts/logs/2025_09_03-18_11_52-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_4_cleaned.pt \
    --pretrained=/home/h3c/lyl/open_clip/models/ViT-B-16/openai/ViT-B-16.pt
   #  --distill-pretrained /home/h3c/lyl/open_clip/src/open_clip_train/logs/2025_08_14-14_24_19-model_ViT-B-16-lr_0.0001-b_900-j_1-p_amp/checkpoints/dist_epoch_28.pt \ 较小的ibm
   #  --resume /home/h3c/lyl/open_clip/scripts/logs/2025_08_19-00_37_53-model_ViT-B-16-lr_1e-06-b_92-j_4-p_amp/checkpoints/epoch_21.pt
   #  --distill-pretrained /home/h3c/lyl/open_clip/scripts/logs/2025_08_29-10_21_37-model_ViT-B-16-lr_0.0001-b_92-j_1-p_amp/checkpoints/dist_epoch_30.pt \ 多层教师

 #少层教师/home/h3c/lyl/open_clip/scripts/logs/2025_09_03-18_11_52-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_4.pt \
 #多层教师/home/h3c/lyl/open_clip/scripts/logs/2025_09_11-10_03_18-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_5.pt \
