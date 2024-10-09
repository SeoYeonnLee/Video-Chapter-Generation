# python3 pretrain_lang_model.py --gpu 0 --epoch 3000 --learn_pe --glove_emb
# python3 pretrain_lang_model.py --gpu 1 --epoch 3000 --learn_pe

# 当前run可能无比较意义，因为数据集总量比之前的更大了
# python3 pretrain_lang_model.py --gpu 0 --model_config --layer_num 8 --batch_size 64 --epoch 3000 --learn_pe


# python3 pretrain_lang_model.py --gpu 0 --epoch 3000 --glove_emb
# python3 pretrain_lang_model.py --gpu 1 --epoch 3000


# python3 pretrain_lang_model.py --gpu 1 --epoch 3000 --lr_decay_type exp --learn_pe --glove_emb 
# python3 pretrain_lang_model.py --gpu 1 --epoch 3000


# run hugface pretrained model
# python3 pretrain_lang_model_hugface.py --gpu 0 --model_type gpt --batch_size 64 --epoch 3000
python3 pretrain_lang_model_hugface.py --gpu 0 --model_type bert --batch_size 64 --epoch 3000

