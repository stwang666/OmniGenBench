#!/bin/bash
#SBATCH --account=brics.u5cs
#SBATCH --partition=workq
#SBATCH --job-name=precompute_struct
#SBATCH --output=precompute_%j.log
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=12:00:00

cd /home/u5cs/stwang.u5cs/OmniGenBench/examples/dingling_te_structure_new20251205_new_encoder/backbone_graphormer_main

source ~/.bashrc
conda activate omnigen_env

export OMP_NUM_THREADS=1

# 预计算训练集的结构信息
# 使用更少的进程数（8个）和更小的批次（3000），避免OOM
echo "=========================================="
echo "开始预计算训练集的结构信息..."
echo "=========================================="
python precompute_structure.py \
    --csv_file train.csv \
    --output_file structure_cache_train.pkl \
    --max_spatial_pos 32 \
    --num_workers 8 \
    --batch_size 3000

if [ $? -ne 0 ]; then
    echo "错误：训练集结构信息预计算失败！"
    exit 1
fi

# 预计算验证集的结构信息
echo ""
echo "=========================================="
echo "开始预计算验证集的结构信息..."
echo "=========================================="
python precompute_structure.py \
    --csv_file valid.csv \
    --output_file structure_cache_valid.pkl \
    --max_spatial_pos 32 \
    --num_workers 8 \
    --batch_size 3000

if [ $? -ne 0 ]; then
    echo "错误：验证集结构信息预计算失败！"
    exit 1
fi

# 预计算测试集的结构信息
echo ""
echo "=========================================="
echo "开始预计算测试集的结构信息..."
echo "=========================================="
python precompute_structure.py \
    --csv_file test.csv \
    --output_file structure_cache_test.pkl \
    --max_spatial_pos 32 \
    --num_workers 8 \
    --batch_size 3000

if [ $? -ne 0 ]; then
    echo "错误：测试集结构信息预计算失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 所有结构信息预计算完成！"
echo "=========================================="
echo "生成的文件："
ls -lh structure_cache_*.pkl

