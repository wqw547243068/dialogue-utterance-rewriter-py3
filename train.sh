CUDA_VISIBLE_DEVICES=2 python run_summarization.py \
    --mode=train \
    --data_path=data/train.txt \
    --vocab_path=data/vocab.txt \
    --log_root=./log \
    --exp_name=extractive \
    --vocab_size=4000 \
    --coverage=0 \
    --batch_size=128 \
    --convert_to_coverage_model=0 \
    --restore_best_model=0 # 初次运行赋值0，否则赋值1 

# 【2020-5-13】多轮对话改写实战，https://cloud.tencent.com/developer/article/1624349
