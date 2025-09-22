import torch
import logging
import os
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
import sys

def main(args):
    # 构造参数对象，模仿main.py的args结构
    args = parse_args(args)
    
    # Initialize CUDA settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Initialize device (模仿main.py的设备初始化)
    device = init_distributed_device(args)
    
    # Setup logging
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(None, args.log_level)
    
    # Create model and transforms (模仿main.py的模型创建)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
    )
    
    # Initialize tokenizer (模仿main.py的tokenizer初始化)
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    
    # Initialize datasets (模仿main.py的数据初始化)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=0,
        tokenizer=tokenizer,
    )
    
    if 'val' not in data:
        logging.error("No validation data found!")
        return
    
    val_dataloader = data['val'].dataloader
    logging.info(f"Validation dataset size: {data['val'].dataloader.num_samples}")
    
    # Process data and save features
    all_image_features = []
    all_text_features = []
    
    model.eval()
    with torch.no_grad():
        for i, (images, texts) in enumerate(val_dataloader):
                
            # Move to device
            images = images.to(device)
            texts = texts.to(device)
            
            # Forward pass
            out_dict = model(image=images, text=texts)
            
            # Collect features
            all_image_features.append(out_dict["image_features"].cpu())
            all_text_features.append(out_dict["text_features"].cpu())
            
            logging.info(f"Processed batch {i+1}")
    
    # Concatenate all features
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # Save features
    image_path = "image_features.pth"
    text_path = "text_features.pth"
    
    torch.save(all_image_features, image_path)
    torch.save(all_text_features, text_path)
    
    logging.info(f"Saved image features shape: {all_image_features.shape} to {image_path}")
    logging.info(f"Saved text features shape: {all_text_features.shape} to {text_path}")

if __name__ == "__main__":
    main(sys.argv[1:])