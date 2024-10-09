# load pretrained model and train for chapter generation task
python3 train_chapter_title_gen.py --model_type bigbird --gpu 0


# export OMP_NUM_THREADS=4
# accelerate launch train_chapter_title_gen_accelerator.py

