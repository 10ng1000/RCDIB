CUDA_VISIBLE_DEVICES=0,1 python -m open_clip_train.main \
 	--val-data="/home/h3c/lyl/data/wds_mscoco_captions/test/{0..4}.tar"  \
   --workers=4 \
    --seed=42 \
    --model=ViT-L-14 \
    --pretrained=/home/h3c/lyl/open_clip/models/ViT-L-14/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin \