"""
test chapter title generation per video
"""


import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from common_utils import set_random_seed
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from data.youtube_chapter_title_dataset import YoutubeChapterTitleDataset
from model.lang import pegasus_hugface


if __name__ == "__main__":
    set_random_seed.use_fix_random_seed()
    device = "cuda:7"
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/chapter_title_gen/chapter_title_hugface_pegasus/batch_64_lr_decay_cosine/checkpoint.pth"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    test_vid_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/test.txt"


    # tokenizer and model
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    model = pegasus_hugface.PegasusHugface(reinit_head=True).to(device)
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint["epoch"]
    best_result = checkpoint["best_result"]
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(checkpoint)
    model.eval()


    # dataset
    max_text_len = 512
    chapter_title_text_len = 30
    dataset = YoutubeChapterTitleDataset(data_file, test_vid_file, tokenizer, max_text_len, chapter_title_text_len)
    # loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=8, num_workers=4)
    loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=1)

    # test on test dataset
    losses = []
    accs = []
    for text_ids, attention_mask, input_decode_ids, decode_attention_mask, target_decode_ids in loader:
        original_text = tokenizer.batch_decode(text_ids)
        print(original_text)

        text_ids = text_ids.to(device)
        attention_mask = attention_mask.to(device)   
        input_decode_ids = input_decode_ids.to(device)
        decode_attention_mask = decode_attention_mask.to(device)
        target_decode_ids = target_decode_ids.to(device)

        gen_text_one_by_one, sentence_logits = model.generate(original_text[0], tokenizer, device)
        print(f"gen_text_one_by_one: {gen_text_one_by_one}")

        # forward the model
        with torch.no_grad():
            # decode_text_ids_dummy = decode_text_ids[:, :1]
            # decode_attention_mask_dummy = decode_attention_mask[:, :1]
            # logits_dummy = model(text_ids, attention_mask, decoder_input_ids=decode_text_ids_dummy, decoder_attention_mask=decode_attention_mask_dummy)
            
            logits = model(text_ids, attention_mask, decoder_input_ids=input_decode_ids, decoder_attention_mask=decode_attention_mask)


        # calculate loss and acc
        mask = torch.nonzero(decode_attention_mask == 1)
        valid_logits = logits[mask[:, 0], mask[:, 1], :]
        valid_targets = target_decode_ids[mask[:, 0], mask[:, 1]]
        loss = F.cross_entropy(valid_logits.view(-1, valid_logits.size(-1)), valid_targets.view(-1))
        
        # acc
        cpu_y = valid_targets.cpu().numpy()
        topk_scores, topk_labels = valid_logits.data.topk(1, 1, True, True)
        topk_ind = topk_labels.squeeze(1).cpu().numpy()

        gt_tokens = tokenizer.convert_ids_to_tokens(cpu_y)
        gt_text = tokenizer.convert_tokens_to_string(gt_tokens)
        gen_tokens = tokenizer.convert_ids_to_tokens(topk_ind)
        gen_text = tokenizer.convert_tokens_to_string(gen_tokens)

        correct = np.sum(topk_ind == cpu_y)
        count = len(cpu_y)
        acc = correct / count

        losses.append(loss.item())
        accs.append(acc)
        print(f"GT: {gt_text}, Gen: {gen_text}")    # the gen_text may be different from gen_text_one_by_one, because input_decode_ids for gen_text is GT text 
        print()

    print(f"avg loss {sum(losses)/len(losses)}, avg acc {sum(accs)/len(accs)}")

