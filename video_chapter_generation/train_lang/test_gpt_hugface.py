import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import OpenAIGPTTokenizer, BertTokenizer
from model.lang import gpt_hugface, bert_hugface
from data.youtube_subtitle_dataset import YoutubeClipSubtitleDatasetForHugFace
from common_utils import language_model_utils


if __name__ == "__main__":
    device = "cuda:3"
    model_type = "gpt"
    batch_size = 64
    lr_decay_type = "cosine"
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/hugface_{model_type}_pretrain/batch_{batch_size}_lr_decay_{lr_decay_type}/pretrain.pth"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"

    # tokenizer and model
    if model_type == "gpt":
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = gpt_hugface.GPTHugface().to(device)
    elif model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = bert_hugface.BertHugface().to(device)
    else:
        raise RuntimeError(f"Unknown model type {model_type}")

    # dataset
    block_size = 50
    dataset = YoutubeClipSubtitleDatasetForHugFace(data_file, model_type, tokenizer, clip_frame_num=8, block_size=block_size)
    
    # loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=64, num_workers=8)
    single_batch_loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=1, num_workers=0)

    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model.eval()

    # test on dataset
    batch_count = 0
    for x, y, attention_mask in single_batch_loader:
        if batch_count > 5:
            break
        x = x.to(device)
        y = y.to(device)
        attention_mask = attention_mask.to(device)  

        # forward the model
        with torch.no_grad():
            logits, loss = model(x, attention_mask, y)

        # convert to sentence
        if model_type == "gpt":
            m_logits = logits.squeeze(0)
            m_logits = language_model_utils.top_k_logits(m_logits, k=3)
            probs = F.softmax(m_logits, dim=-1).squeeze(0)
            # sample from the distribution or take the most likely
            sentence_ids = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()
            gt_y = y.squeeze(0).cpu().numpy()
            sentence_ids = list(sentence_ids[gt_y != -1])
            gt_y = list(gt_y[gt_y != -1])

            first_token_id = int(x[0, 0].item())
            sentence_ids = [first_token_id] + sentence_ids
            gt_y = [first_token_id] + gt_y

            gen_sentence = tokenizer._decode(sentence_ids)
            gt_sentence = tokenizer._decode(gt_y)

            print(f"Gen: {gen_sentence}")
            print(f"GT: {gt_sentence}")
        else:
            m_logits = logits.squeeze(0)
            m_logits = language_model_utils.top_k_logits(m_logits, k=1)
            probs = F.softmax(m_logits, dim=-1).squeeze(0)
            sentence_ids = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()
            gt_y = y.squeeze(0).cpu().numpy()
            masked_sentence_ids = list(sentence_ids[gt_y != -1])
            masked_gt_y = list(gt_y[gt_y != -1])

            masked_indices = gt_y != -1
            original_x = x.squeeze(0).cpu().numpy()
            gt_x = np.array(original_x)
            gt_x[masked_indices] = gt_y[masked_indices]
            gen_x = np.array(original_x)
            gen_x[masked_indices] = sentence_ids[masked_indices]

            original_sentence = tokenizer._decode(original_x)
            gen_sentence = tokenizer._decode(gen_x)
            gt_sentence = tokenizer._decode(gt_x)

            print(f"Original {original_sentence}")
            print(f"GT: {gt_sentence}")
            print(f"Gen: {gen_sentence}")
            
        # acc
        mask = torch.nonzero(y != -1)
        valid_logits = logits[mask[:, 0], mask[:, 1], :]
        valid_y = y[mask[:, 0], mask[:, 1]]

        cpu_y = valid_y.cpu().numpy()
        topk_scores, topk_labels = valid_logits.data.topk(1, 1, True, True)
        topk_ind = topk_labels.squeeze(1).cpu().numpy()
        correct = np.sum(topk_ind == cpu_y)
        count = len(cpu_y)

        print(f"acc {correct}/{count}, {correct/count}\n\n")

        batch_count += 1


    # generate self-defined sentence
    if model_type == "gpt":
        print("\n\nGenerate self-defined sentence...")
        generate_num_each_init_text = 2
        init_texts = ["let's get cooking the first", "so the first game of the day"]
        for init_text in init_texts:
            input_ids = tokenizer.encode(init_text, return_tensors="pt").to(device)
            for i in range(generate_num_each_init_text):
                embs, sentence_ids = language_model_utils.token_idx_sample(model, input_ids, block_size, temperature=1.0, sample=True, top_k=10)
                gen_sentence = tokenizer._decode(sentence_ids)
                print(init_text + " * " + gen_sentence)
