import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.lang.gpt import GPT, GPTConfig
from data.youtube_subtitle_dataset import YoutubeClipSubtitleGloveDataset, YoutubeClipSubtitleDataset
from common_utils import language_model_utils
from data.common_utils import text_decontracted



if __name__ == "__main__":
    device = "cuda:3"
    use_glove_emb = False
    learnable_pos_emb = True
    lr_decay_type = "cosine"
    ckpt_path = f"/opt/tiger/video_chapter_generation/checkpoint/gpt_pretrain_using_glove/layer12_batch64_learn_pe_{learnable_pos_emb}_glove_emb_{use_glove_emb}_lr_decay_{lr_decay_type}/gpt_pretrain.pth"
    data_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv"
    glove_pickle_file = "/opt/tiger/video_chapter_generation/glove.840B.300d.pickle"

    if use_glove_emb:
        vocab_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/vocab_in_glove.txt"
        dataset = YoutubeClipSubtitleGloveDataset(data_file, glove_pickle_file, vocab_file, clip_frame_num=8, max_text_len=50)
    else:
        vocab_file = "/opt/tiger/video_chapter_youtube_dataset/dataset/vocab.txt"
        dataset = YoutubeClipSubtitleDataset(data_file, vocab_file, clip_frame_num=8, max_text_len=50)
    # loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=64, num_workers=8)
    single_batch_loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=1, num_workers=0)

    vocab_size = dataset.vocab_size
    block_size = dataset.max_text_len
    # using glove embedding
    mconf = GPTConfig(vocab_size=vocab_size, block_size=block_size, using_pretrained_embed=use_glove_emb, learnable_pos_emb=learnable_pos_emb, n_layer=12, n_head=10, n_embd=300)
    model = GPT(mconf)

    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model.eval()

    # test on dataset
    batch_count = 0
    for x, y in single_batch_loader:
        if batch_count > 5:
            break
        # place data on the correct device
        x = x.to(device)
        y = y.to(device)

        # forward the model
        with torch.no_grad():
            logits, loss = model(x, y)

        # convert to sentence
        m_logits = logits.squeeze(0)
        m_logits = language_model_utils.top_k_logits(m_logits, k=3)
        probs = F.softmax(m_logits, dim=-1).squeeze(0)
        # sample from the distribution or take the most likely
        sentence_ids = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()
        gt_y = y.squeeze(0).cpu().numpy()
        sentence_ids = sentence_ids[gt_y != -1]
        gt_y = gt_y[gt_y != -1]
        if use_glove_emb:
            first_word = " "
        else:
            first_word = dataset.id2token[int(x[0, 0].item())]
        completion = ' '.join([dataset.id2token[int(i)] for i in sentence_ids])
        gt_completion = ' '.join([dataset.id2token[int(i)] for i in gt_y])
        gen_sentence = first_word + " " + completion
        gt_sentence = first_word + " " + gt_completion
        print(f"Gen: {gen_sentence}")
        print(f"GT: {gt_sentence}")

        # acc
        mask = torch.nonzero(y != -1)
        valid_logits = logits[mask[:, 0], mask[:, 1], :]
        valid_y = y[mask[:, 0], mask[:, 1]]

        cpu_y = valid_y.cpu().numpy()
        topk_scores, topk_labels = valid_logits.data.topk(1, 1, True, True)
        topk_ind = topk_labels.squeeze(1).cpu().numpy()
        correct = np.sum(topk_ind == cpu_y)
        count = len(cpu_y)

        print(f"acc {correct}/{count}, {correct/count}")

        batch_count += 1



    # generate self-defined sentence
    print("generate self-defined sentence")
    generate_num_each_init_text = 2
    init_texts = ["let's get cooking the first", "so the first game of the day"]
    for init_text in init_texts:
        init_text = text_decontracted(init_text)
        context = init_text.split(" ")
        for i in range(generate_num_each_init_text):
            if use_glove_emb:
                x = torch.tensor([dataset.token2embedding[s] for s in context], dtype=torch.float32)[None,...].to(device)
                embs, sentence_tokens = language_model_utils.token_embedding_sample(model, x, block_size, device, dataset, temperature=1.0, sample=True, top_k=10)
                completion = ' '.join(sentence_tokens)
            else:
                x = torch.tensor([dataset.token2id[s] for s in context], dtype=torch.long)[None,...].to(device)
                embs, sentence_ids = language_model_utils.token_idx_sample(model, x, block_size, temperature=1.0, sample=True, top_k=10)
                completion = ' '.join([dataset.id2token[int(i)] for i in sentence_ids])
            
            print(init_text + " * " + completion)

