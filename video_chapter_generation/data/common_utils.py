import pandas as pd
import re
import pickle


def parse_csv_to_list(csv_file):
    TIMESTAMP_DELIMITER = "%^&*"
    data = pd.read_csv(csv_file)
    vids = list(data["videoId"].values)
    titles = list(data["title"].values)
    durations = list(data["duration"].values)
    timestamps = list(data["timestamp"].values)
    timestamps = [x.split(TIMESTAMP_DELIMITER) for x in timestamps]

    return vids, titles, durations, timestamps

def clean_str(s):
    """
    Remove all special char at the beginning and the end.
    Use to clean chapter title string
    """
    start_idx = 0
    for i in range(len(s)):
        if s[i].isalnum():
            start_idx = i
            break

    end_idx = len(s)
    for i in reversed(range(len(s))):
        if s[i].isalnum():
            end_idx = i + 1
            break
    
    return s[start_idx : end_idx]


def extract_timestamp(s):
    r = re.search("\d{2}:\d{2}:\d{2}", s)
    if r:
        si, ei = r.regs[0]
    else:
        r = re.search("\d{1}:\d{2}:\d{2}", s)
        if r:
            si, ei = r.regs[0]
        else:
            r = re.search("\d{2}:\d{2}", s)
            if r:
                si, ei = r.regs[0]
            else:
                r = re.search("\d{1}:\d{2}", s)
                if r:
                    si, ei = r.regs[0]
                else:
                    return "", -1, -1, -1

    timestamp = s[si:ei]
    ts = timestamp.split(":")
    ts.reverse()
    sec = 0
    for i in range(len(ts)):
        if i == 0:
            sec += int(ts[i])
        elif i == 1:
            sec += int(ts[i]) * 60
        elif i == 2:
            sec += int(ts[i]) * 3600

    return s[si:ei], sec, si, ei


def extract_first_timestamp(s):
    t, sec, si, ei = extract_timestamp(s)
    min_sec = sec
    description = s[:si] + s[ei:]

    while sec != -1:
        t, sec, si, ei = extract_timestamp(description)
        if sec != -1:
            if min_sec > sec:
                min_sec = sec
            description = description[:si] + description[ei:]
    
    return min_sec, description



def remove_timestamp(s):
    r = re.search("\d{2}:\d{2}:\d{2}", s)
    if r:
        si, ei = r.regs[0]
    else:
        r = re.search("\d{1}:\d{2}:\d{2}", s)
        if r:
            si, ei = r.regs[0]
        else:
            r = re.search("\d{2}:\d{2}", s)
            if r:
                si, ei = r.regs[0]
            else:
                r = re.search("\d{1}:\d{2}", s)
                if r:
                    si, ei = r.regs[0]
                else:
                    return s
    
    ss = s[:si] + s[ei:]
    ss = [x for x in ss.split(" ") if len(x) > 0]
    ss = " ".join(ss)
    return ss


def load_glove_model_from_txt(txt_file):
    print("Loading Glove Model")
    f = open(txt_file,'r')
    gloveModel = {}
    for line in f:
        try:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        except Exception as e:
            continue
    print(len(gloveModel)," words loaded!")
    return gloveModel


def load_glove_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as handle:
        token2embedding = pickle.load(handle)
    return token2embedding


def text_decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"t\'s", "t us", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase



if __name__ == "__main__":
    s0 = "Stack 2 7:08-11:31"
    # s0 = "Stack 2 45646sss"
    min_sec, description = extract_first_timestamp(s0)
    print()

    # s0 = "0:46 introduction"
    # s = "2:25 distilled vs tap water"
    # new_s = remove_timestamp(s)
    # print(new_s)


    # import time
    # glove_txt_file = "/opt/tiger/video_chapter_generation/glove.840B.300d.txt"
    # glove_pickle_file = "/opt/tiger/video_chapter_generation/glove.840B.300d.pickle"

    # # st = time.time()
    # # token2embedding = load_glove_model_from_txt(glove_txt_file)
    # # et = time.time()
    # # print(f"load cost {et - st}")

    # # with open(glove_pickle_file, 'wb') as handle:
    # #     pickle.dump(token2embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # st = time.time()
    # with open(glove_pickle_file, 'rb') as handle:
    #     token2embedding_2 = pickle.load(handle)
    # et = time.time()
    # print(f"load cost {et - st}")

    # print(token2embedding_2["use"])