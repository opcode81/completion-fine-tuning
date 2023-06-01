DATE=$(date +'%Y%m%d-%H%M')
python train.py \
        --model_path="bigcode/santacoder" \
        --dataset_name="bigcode/the-stack-dedup" \
        --subset="data/${LANGUAGE}" \
        --data_column "content" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 30000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_warmup_steps 500 \
        --eval_freq 500 \
        --save_freq 50 \
        --log_freq 1 \
        --num_workers="$(nproc)" \
        --no_fp16 \
        --fim_rate 0.5 \
        --fim_spm_rate 0.5 \
  2>&1 | tee train_$LANGUAGE_$DATE.log