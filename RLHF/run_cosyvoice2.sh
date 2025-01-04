#!/bin/bash

# 创建日志目录
mkdir -p logs

# 运行时记录开始时间
echo "Starting jobs at $(date)" > logs/main.log

CUDA_VISIBLE_DEVICES=0 python inference_samples.py --start 0 --end 1500 > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python inference_samples.py --start 1500 --end 3000 > logs/gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python inference_samples.py --start 3000 --end 4500 > logs/gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python inference_samples.py --start 4500 --end 6000 > logs/gpu3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python inference_samples.py --start 6000 --end 7500 > logs/gpu4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python inference_samples.py --start 7500 --end 9000 > logs/gpu5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python inference_samples.py --start 9000 --end 10500 > logs/gpu6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python inference_samples.py --start 10500 --end 12000 > logs/gpu7.log 2>&1 &

wait

# 记录结束时间
echo "All jobs completed at $(date)" >> logs/main.log