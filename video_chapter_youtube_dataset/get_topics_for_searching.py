"""
crawl wikihow to get 'how to xxx' subjects which can be treated as queries in youtube

"""

import glob
import json
import requests
from bs4 import BeautifulSoup

wiki_how_website = "https://www.wikihow.com/"
subjects = [
"Category:Arts-and-Entertainment",
"Category:Cars-%26-Other-Vehicles",
"Category:Computers-and-Electronics",
"Category:Education-and-Communications",
"Category:Family-Life",
"Category:Finance-and-Business",
"Category:Food-and-Entertaining",
"Category:Health",
"Category:Hobbies-and-Crafts",
"Category:Holidays-and-Traditions",
"Category:Home-and-Garden",
"Category:Personal-Care-and-Style",
"Category:Pets-and-Animals",
"Category:Sports-and-Fitness",
"Category:Travel",
"Category:Work-World",
"Category:Youth",
]


def parse_wikihow_webpage_to_get_queries():
    category2query = dict()
    query = []
    for subject in subjects:
        url = wiki_how_website + subject
        resp = requests.get(url)
        qs = []
        if resp.status_code == 200:
            html_page = resp.content
            html_page = html_page.decode("utf-8")
            soup = BeautifulSoup(html_page)
            mydivs = soup.find_all("div", {"class": "responsive_thumb_title"})
            for div in mydivs:
                text = div.text
                text = text.replace("\n", " ")
                t = [x for x in text.split(" ") if len(x) > 0]
                text = " ".join(t)
                query.append(text)
                qs.append(text)
            
            category2query[subject] = qs
            print(f"subject {subject}, get {len(query)} queries")

    # with open("./wikihow_query.txt", "w") as f:
    #     for q in query:
    #         f.write(q+"\n")

    return category2query


def find_query_belong_category():
    category2query = parse_wikihow_webpage_to_get_queries()

    with open("wikihow_query_total.txt", "r") as f:
        query = [x.strip() for x in f.readlines()]
    
    category_count = {"unknown": 0}
    query2category = {}
    for q in query:
        find_a_category = False
        for k, v in category2query.items():
            if q in v:
                find_a_category = True
                if k in category_count:
                    category_count[k] += 1
                else:
                    category_count[k] = 1
                query2category[q] = k
                break
        
        if not find_a_category:
            category_count["unknown"] += 1
            query2category[q] = "unknown"

    print(category_count)
    
    # get all valid vids
    with open("dataset/new_train.txt", "r") as f:
        train_vids = [x.strip() for x in f.readlines()]
    with open("dataset/new_validation.txt", "r") as f:
        valid_vids = [x.strip() for x in f.readlines()]
    with open("dataset/new_test.txt", "r") as f:
        test_vids = [x.strip() for x in f.readlines()]
    total_vids = train_vids + valid_vids + test_vids

    # all_vid2category
    subtitle_files = glob.glob("dataset/*/*.json")
    all_vid2category = dict()
    for f in subtitle_files:
        tokens = f.split("/")
        query = tokens[1]
        vid = tokens[2][9:-5]

        if query in query2category:
            category = query2category[query]
        else:
            category = "unknown"
        all_vid2category[vid] = category
    
    # valid vid 2 category
    valid_vid2category = dict()
    for vid in total_vids:
        valid_vid2category[vid] = all_vid2category[vid]
    
    # category 2 valid vid
    category2valid_vid = dict()
    for k, v in valid_vid2category.items():
        if v in category2valid_vid:
            category2valid_vid[v].append(k)
        else:
            category2valid_vid[v] = [k]
    
    # print count
    for k, v in category2valid_vid.items():
        print(f"{k}, {len(v)}")
    
    with open("dataset/category2total_vid.json", "w") as f:
        json.dump(category2valid_vid, f)


if __name__ == "__main__":
    # parse_wikihow_webpage_to_get_queries()
    find_query_belong_category()