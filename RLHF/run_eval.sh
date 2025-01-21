save_root="/storage/zhangxueyao/workspace/CosyVoice/results"

# 创建日志目录
log_dir="${save_root}/log_inference"
mkdir -p $log_dir

model_name="cosyvoice2"

eval_setting="seedtts"
CUDA_VISIBLE_DEVICES=7 python inference_for_eval.py \
    --save_root $save_root \
    --model_name $model_name \
    --eval_setting $eval_setting > "${log_dir}/${model_name}_${eval_setting}.log" 2>&1 &

eval_setting="hard"
CUDA_VISIBLE_DEVICES=6 python inference_for_eval.py \
    --save_root $save_root \
    --model_name $model_name \
    --eval_setting $eval_setting > "${log_dir}/${model_name}_${eval_setting}.log" 2>&1 &

eval_setting="crosslingual"
CUDA_VISIBLE_DEVICES=5 python inference_for_eval.py \
    --save_root $save_root \
    --model_name $model_name \
    --eval_setting $eval_setting > "${log_dir}/${model_name}_${eval_setting}.log" 2>&1 &

eval_setting="codeswitching"
CUDA_VISIBLE_DEVICES=4 python inference_for_eval.py \
    --save_root $save_root \
    --model_name $model_name \
    --eval_setting $eval_setting > "${log_dir}/${model_name}_${eval_setting}.log" 2>&1 &

# 等待所有后台进程完成
wait

echo "所有推理任务已完成，日志文件保存在 ${log_dir} 目录下"