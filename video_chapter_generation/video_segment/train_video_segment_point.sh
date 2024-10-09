# train text mode for chapter segment classification
# python3 train_video_segment_point.py --data_mode text --model_type bert --gpu 7
# python3 train_video_segment_point.py --data_mode text --model_type bert --clip_frame_num 16 --gpu 7


# train image mode for chapter segment classification
# python3 train_video_segment_point.py --data_mode image --model_type r50tsm --gpu 0
# python3 train_video_segment_point.py --data_mode image --model_type r50 --gpu 0
# python3 train_video_segment_point.py --data_mode image --model_type r50tsm --clip_frame_num 16 --gpu 0
# python3 train_video_segment_point.py --data_mode image --model_type r50 --clip_frame_num 16 --gpu 0


# train image + text mode for chapter segment classification
# python3 train_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --gpu 0
# python3 train_video_segment_point.py --data_mode all --model_type two_stream --head_type attn --gpu 0


python3 train_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --clip_frame_num 8 --epoch 271 --gpu 0 
python3 train_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --clip_frame_num 12 --epoch 271 --gpu 0
# python3 train_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --clip_frame_num 16 --epoch 280 --gpu 0
python3 train_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --clip_frame_num 20 --epoch 271 --gpu 0
python3 train_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --clip_frame_num 24 --epoch 271 --gpu 0


