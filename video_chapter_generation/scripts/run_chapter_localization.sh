#!/bin/bash

####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: AFQ+EWGS
# Bit-width: T2, T3, T4, W2A2, W3A3, W4A4
####################################################################################


set -e
# make the script exit with an error whenever an error occurs (and is not explicitly handled).

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


METHOD_TYPE=$1
echo $METHOD_TYPE

### AFQ + use student quant params
if [ $METHOD_TYPE == "test/" ]
then
    python3 train_video_segment_ddp.py --gpu_id '0,1' \
                    --data_mode 'all' \
                    --model_type 'two_stream' \
                    --clip_frame_num 16 \
                    --epoch 270 \
                    --batch_size 4 \
                    --val_batch_size 32 \
                    --lr_decay_type 'cosine' \
                    --head_type 'cross_attn' \
                    --window_size 1 \
                    --num_workers 8 \
                    --grad_accum_step 4\
                    --max_text_len 100 \
                    --start_epoch 0 \
                    --learning_rate 2e-6 \
                    --train_vid_file './dataset/debugging_train.txt' \
                    --test_vid_file './dataset/debugging_val.txt' \
                    --img_dir './dataset/youtube_video_frame_dataset' \
                    --data_file './dataset/all_in_one_with_subtitle_final.csv' \
                    --test_clips_json './dataset/debugging_val_clips_clip_frame_num_16.json' \
                    --lang_pretrain_ckpt_path './checkpoint/hugface_bert_pretrain/batch_64_lr_decay_cosine_train_test_split/pretrain.pth'\
                    --ckpt_path './checkpoint/chapter_localization/'$METHOD_TYPE




fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
total_time=$(( $end - $start ))

echo "RESULT, method type: $METHOD_TYPE, time: $total_time, starting time: $start_fmt"