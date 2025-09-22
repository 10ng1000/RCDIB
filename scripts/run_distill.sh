torchrun --nproc_per_node 2 -m open_clip_train.main \
   --report-to tensorboard \
    --save-frequency 1 \
 	--train-data="/home/h3c/lyl/data/flickr30k/train/{0..4}.tar::/home/h3c/lyl/data/wds_mscoco_captions/train/{0..13}.tar" \
    --val-data="/home/h3c/lyl/data/wds_mscoco_captions/test/{0..4}.tar" \
    --warmup 500 \
    --batch-size 64 \
    --lr=1e-6 \
    --wd=0.1 \
    --epochs=30 \
    --workers=4 \
    --seed=42 \
    --model=ViT-B-16 \
    --train-num-samples 111783 \
    --distill-model=ViT-L-14 \
    --enable-ibm-module \
    --distill-pretrained /home/h3c/lyl/open_clip/scripts/logs/2025_09_16-16_27_04-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_1.pt \
    --pretrained=/home/h3c/lyl/open_clip/scripts/logs/2025_09_15-20_13_46-model_RN50-lr_1e-06-b_256-j_4-p_amp/checkpoints/epoch_3.pt \

    # 不归一化版本
   #  --distill-pretrained=/home/h3c/lyl/open_clip/scripts/logs/2025_09_17-09_19_07-model_RN50-lr_0.001-b_256-j_4-p_amp/checkpoints/epoch_5.pt\
   #  --pretrained=/home/h3c/lyl/open_clip/scripts/logs/2025_09_17-09_24_41-model_RN50-lr_0.001-b_256-j_4-p_amp/checkpoints/epoch_5.pt

    # 归一化版本

# torchrun --nproc_per_node 2 -m open_clip_train.main \
#    --report-to tensorboard \
#     --save-frequency 1 \
#  	--train-data="/home/h3c/lyl/data/flickr30k/train/{0..4}.tar::/home/h3c/lyl/data/wds_mscoco_captions/train/{0..13}.tar" \
#     --val-data="/home/h3c/lyl/data/wds_mscoco_captions/test/{0..4}.tar" \
#     --warmup 800 \
#     --batch-size 64 \
#     --lr=1e-6 \
#     --wd=0.1 \
#     --epochs=20 \
#     --workers=4 \
#     --seed=42 \
#     --model=ViT-B-32 \
#     --train-num-samples 111783 \
#     --distill-model=ViT-L-14 \
#     --enable-ibm-module \
#     --distill-pretrained /home/h3c/lyl/open_clip/scripts/logs/2025_09_16-16_27_04-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_1.pt \
#     --pretrained=/home/h3c/lyl/open_clip/models/ViT-B-32/openai/ViT-B-32.pt \

# torchrun --nproc_per_node 2 -m open_clip_train.main \
#    --report-to tensorboard \
#     --save-frequency 1 \
#  	--train-data="/home/h3c/lyl/data/flickr30k/train/{0..4}.tar::/home/h3c/lyl/data/wds_mscoco_captions/train/{0..13}.tar" \
#     --val-data="/home/h3c/lyl/data/wds_mscoco_captions/test/{0..4}.tar" \
#     --warmup 800 \
#     --batch-size 64 \
#     --lr=1e-6 \
#     --wd=0.1 \
#     --epochs=20 \
#     --workers=4 \
#     --seed=42 \
#     --model=RN50 \
#     --train-num-samples 111783 \
#     --distill-model=ViT-L-14 \
#     --enable-ibm-module \
#     --distill-pretrained /home/h3c/lyl/open_clip/scripts/logs/2025_09_16-16_27_04-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_1.pt \
#     --pretrained=/home/h3c/lyl/open_clip/models/RN50/RN50.pt \

# torchrun --nproc_per_node 2 -m open_clip_train.main \
#    --report-to tensorboard \
#     --save-frequency 1 \
#  	 --train-data="/home/h3c/lyl/data/flickr30k/train/{0..4}.tar::/home/h3c/lyl/data/wds_mscoco_captions/train/{0..13}.tar" \
#     --val-data="/home/h3c/lyl/data/wds_mscoco_captions/test/{0..4}.tar" \
#     --warmup 800 \
#     --batch-size 64 \
#     --lr=1e-6 \
#     --wd=0.1 \
#     --epochs=20 \
#     --workers=4 \
#     --seed=42 \
#     --model=RN101 \
#     --train-num-samples 111783 \
#     --distill-model=ViT-L-14 \
#     --enable-ibm-module \
#     --distill-pretrained /home/h3c/lyl/open_clip/scripts/logs/2025_09_16-16_27_04-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_1.pt \
#     --pretrained=/home/h3c/lyl/open_clip/models/RN101/RN101.pt \

# torchrun --nproc_per_node 2 -m open_clip_train.main \
#    --report-to tensorboard \
#     --save-frequency 1 \
#  	--train-data="/home/h3c/lyl/data/flickr30k/train/{0..4}.tar::/home/h3c/lyl/data/wds_mscoco_captions/train/{0..13}.tar" \
#     --val-data="/home/h3c/lyl/data/wds_mscoco_captions/test/{0..4}.tar" \
#     --warmup 800 \
#     --batch-size 64 \
#     --lr=1e-6 \
#     --wd=0.1 \
#     --epochs=20 \
#     --workers=4 \
#     --seed=42 \
#     --model=ViT-B-16 \
#     --train-num-samples 111783 \
#     --distill-model=ViT-L-14 \
#     --distill-pretrained /home/h3c/lyl/open_clip/scripts/logs/2025_09_16-16_27_04-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_1.pt \
#     --pretrained=/home/h3c/lyl/open_clip/scripts/logs/2025_09_15-20_13_46-model_RN50-lr_1e-06-b_256-j_4-p_amp/checkpoints/epoch_3.pt \



   #  --distill-pretrained /home/h3c/lyl/open_clip/src/open_clip_train/logs/2025_08_14-14_24_19-model_ViT-B-16-lr_0.0001-b_900-j_1-p_amp/checkpoints/dist_epoch_28.pt \ 较小的ibm
   #  --resume /home/h3c/lyl/open_clip/scripts/logs/2025_08_19-00_37_53-model_ViT-B-16-lr_1e-06-b_92-j_4-p_amp/checkpoints/epoch_21.pt
   #  --distill-pretrained /home/h3c/lyl/open_clip/scripts/logs/2025_08_29-10_21_37-model_ViT-B-16-lr_0.0001-b_92-j_1-p_amp/checkpoints/dist_epoch_30.pt \ 多层教师

 #少层教师/home/h3c/lyl/open_clip/scripts/logs/2025_09_03-18_11_52-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_4.pt \
 #多层教师/home/h3c/lyl/open_clip/scripts/logs/2025_09_11-10_03_18-model_RN50-lr_1e-06-b_16-j_4-p_amp/checkpoints/epoch_5.pt \
