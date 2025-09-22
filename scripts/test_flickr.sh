CUDA_VISIBLE_DEVICES=0,1 python -m open_clip_train.main \
 	--val-data="/home/h3c/lyl/data/flickr30k/test/{0..4}.tar"  \
   --workers=1 \
   --seed=42 \
   --model=ViT-B-16 \
   --pretrained=/home/h3c/lyl/open_clip/scripts/logs/2025_09_17-03_13_00-model_ViT-B-16-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_20.pt
   #B32   --pretrained=/home/h3c/lyl/open_clip/scripts/logs/2025_09_16-21_25_37-model_ViT-B-32-lr_1e-06-b_64-j_4-p_amp/checkpoints/epoch_20.pt