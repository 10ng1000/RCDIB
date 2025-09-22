#!/bin/bash

# 批量移除IBM模块的Shell脚本
# 使用方法:
# ./batch_remove_ibm.sh /path/to/models/directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOVE_IBM_SCRIPT="$SCRIPT_DIR/remove_ibm_from_checkpoint.py"

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <模型目录路径> [输出目录路径]"
    echo "示例: $0 /home/h3c/lyl/open_clip/models/ViT-B-16"
    echo "示例: $0 /home/h3c/lyl/open_clip/models/ViT-B-16 /home/h3c/lyl/open_clip/models/ViT-B-16-clean"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-${INPUT_DIR}-clean}"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "$REMOVE_IBM_SCRIPT" ]; then
    echo "错误: 找不到Python脚本: $REMOVE_IBM_SCRIPT"
    exit 1
fi

echo "开始处理目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "----------------------------------------"

# 首先验证是否有IBM模块
echo "正在验证检查点文件..."
python3 "$REMOVE_IBM_SCRIPT" --input_dir "$INPUT_DIR" --validate_only

echo ""
echo "开始移除IBM模块..."

# 执行移除操作
python3 "$REMOVE_IBM_SCRIPT" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --force

if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "成功完成! 清理后的模型保存在: $OUTPUT_DIR"
    
    # 显示文件大小对比
    echo ""
    echo "文件大小对比:"
    echo "原始文件:"
    du -sh "$INPUT_DIR"/*.pt 2>/dev/null | head -5
    echo ""
    echo "清理后文件:"
    du -sh "$OUTPUT_DIR"/*.pt 2>/dev/null | head -5
else
    echo "处理失败!"
    exit 1
fi
