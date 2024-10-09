
python3 test_chapter_title_gen_vision_emb.py --data_mode all --location_type gt --model_type pegasus --fusion_type cross_attn --gpu 0
python3 test_chapter_title_gen_vision_emb.py --data_mode easy --location_type gt --model_type pegasus --fusion_type cross_attn --gpu 0
python3 test_chapter_title_gen_vision_emb.py --data_mode hard --location_type gt --model_type pegasus --fusion_type cross_attn --gpu 0

python3 test_chapter_title_gen_vision_emb.py --data_mode all --location_type pred --model_type pegasus --fusion_type cross_attn --gpu 0
python3 test_chapter_title_gen_vision_emb.py --data_mode easy --location_type pred --model_type pegasus --fusion_type cross_attn --gpu 0
python3 test_chapter_title_gen_vision_emb.py --data_mode hard --location_type pred --model_type pegasus --fusion_type cross_attn --gpu 0



# python3 test_chapter_title_gen_vision_emb.py --data_mode all --location_type gt --model_type pegasus --fusion_type mlp --gpu 0
# python3 test_chapter_title_gen_vision_emb.py --data_mode easy --location_type gt --model_type pegasus --fusion_type mlp --gpu 0
# python3 test_chapter_title_gen_vision_emb.py --data_mode hard --location_type gt --model_type pegasus --fusion_type mlp --gpu 0

# python3 test_chapter_title_gen_vision_emb.py --data_mode all --location_type pred --model_type pegasus --fusion_type mlp --gpu 0
# python3 test_chapter_title_gen_vision_emb.py --data_mode easy --location_type pred --model_type pegasus --fusion_type mlp --gpu 0
# python3 test_chapter_title_gen_vision_emb.py --data_mode hard --location_type pred --model_type pegasus --fusion_type mlp --gpu 0