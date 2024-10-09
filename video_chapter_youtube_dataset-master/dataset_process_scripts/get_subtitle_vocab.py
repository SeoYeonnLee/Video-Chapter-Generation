"""
Create vocab.txt from all subtitle files

"""

import pandas as pd
import re
import json, pickle
import os, glob


def text_decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def load_glove_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as handle:
        token2embedding = pickle.load(handle)
    return token2embedding


if __name__ == "__main__":
    data = pd.read_csv("/opt/tiger/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle.csv")
    vids = list(data["videoId"].values)

    vocab_dict = dict()
    all_subtitle_files = glob.glob("/opt/tiger/video_chapter_youtube_dataset/dataset/*/subtitle_*.json")
    save_vocab_file_path = "/opt/tiger/video_chapter_youtube_dataset/dataset/vocab_large.txt"
    save_glove_vocab_file_path = "/opt/tiger/video_chapter_youtube_dataset/dataset/vocab_in_glove_large.txt"

    vid2subtitle_files = dict()
    for subtitle_file in all_subtitle_files:
        filename = os.path.basename(subtitle_file)
        vid = filename.split(".")[0][9:]
        vid2subtitle_files[vid] = subtitle_file

    for i, vid in enumerate(vids):
        subtitle_file = vid2subtitle_files[vid]
        print(f"load {i}/{len(vids)} subtitle file...")
        with open(subtitle_file, "r") as f:
            sub_list = json.load(f)

        text = ""
        for sub_item in sub_list:
            te = sub_item["text"]
            if len(text) == 0:
                text = te
            else:
                text = text + " " + te
        text = text.lower()
        text = text_decontracted(text)

        vocab_list = text.split(" ")
        for vocab in vocab_list:
            if vocab in vocab_dict:
                vocab_dict[vocab][1] += 1
            if vocab not in vocab_dict and len(vocab) > 0:
                vocab_dict[vocab] = [len(vocab_dict), 1]

    # rule out low frequency token
    high_freq_vocav_dict = dict()
    for k, v in vocab_dict.items():
        if v[1] < 5:
            continue
        high_freq_vocav_dict[k] = v


    # try to match with glove
    glove_pickle_file = "/opt/tiger/video_chapter_generation/glove.840B.300d.pickle"
    token2embedding = load_glove_from_pickle(glove_pickle_file)

    all_keys = set(list(vocab_dict.keys()))
    glove_all_keys = set(list(token2embedding.keys()))
    known_keys = list(all_keys.intersection(glove_all_keys))
    unknown_keys = list(all_keys - glove_all_keys)

    print(f"the number of known glove tokens {len(known_keys)}")
    print(f"the number of unknown glove tokens {len(unknown_keys)}")


    all_keys_high_freq = set(list(high_freq_vocav_dict))
    high_freq_known_keys = list(all_keys_high_freq.intersection(glove_all_keys))
    high_freq_unknown_keys = list(all_keys_high_freq - glove_all_keys)

    print(f"the number of known glove high frequency tokens {len(high_freq_known_keys)}")
    print(f"the number of unknown glove high frequency tokens {len(high_freq_unknown_keys)}")


    with open(save_glove_vocab_file_path, "w") as f:
        for i, k in enumerate(known_keys):
            f.write(f"{k}\n")


    with open(save_vocab_file_path, "w") as f:
        f.write("[**NULL**]\n")
        for i, k in enumerate(all_keys):
            f.write(f"{k}\n")
