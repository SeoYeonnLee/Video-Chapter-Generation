# # test text mode for chapter segment classification
# python3 test_video_segment_point.py --data_mode text --model_type bert --clip_frame_num 16 --data_type easy --gpu 7
# python3 test_video_segment_point.py --data_mode text --model_type bert --clip_frame_num 16 --data_type hard --gpu 7
# python3 test_video_segment_point.py --data_mode text --model_type bert --clip_frame_num 16 --data_type all --gpu 7


# # test image mode for chapter segment classification
# python3 test_video_segment_point.py --data_mode image --model_type r50tsm --clip_frame_num 16 --data_type easy --gpu 7
# python3 test_video_segment_point.py --data_mode image --model_type r50tsm --clip_frame_num 16 --data_type hard --gpu 7
# python3 test_video_segment_point.py --data_mode image --model_type r50tsm --clip_frame_num 16 --data_type all --gpu 7

# # python3 test_video_segment_point.py --data_mode image --model_type r50 --clip_frame_num 16 --data_type easy --gpu 7
# # python3 test_video_segment_point.py --data_mode image --model_type r50 --clip_frame_num 16 --data_type hard --gpu 7
# python3 test_video_segment_point.py --data_mode image --model_type r50 --clip_frame_num 16 --data_type all --gpu 7


# # test image + text mode for chapter segment classification with mlp head
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --clip_frame_num 16 --data_type easy --gpu 7
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --clip_frame_num 16 --data_type hard --gpu 7
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --clip_frame_num 16 --data_type all --gpu 7

# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type attn --data_type all --gpu 7


# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --data_type all --clip_frame_num 6 --gpu 7
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --data_type all --clip_frame_num 8 --gpu 7
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --data_type all --clip_frame_num 10 --gpu 7
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --data_type all --clip_frame_num 12 --gpu 7
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --data_type all --clip_frame_num 14 --gpu 7
python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --data_type all --clip_frame_num 16 --gpu 7
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --data_type all --clip_frame_num 20 --gpu 7
# python3 test_video_segment_point.py --data_mode all --model_type two_stream --head_type mlp --data_type all --clip_frame_num 24 --gpu 7


