import glob
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list


if __name__ == "__main__":
    urls = []
    all_data_file = glob.glob("../dataset/*/data.csv")
    for data_file in all_data_file:
        vids, titles, timestamps = parse_csv_to_list(data_file)
        vids = vids[:5]
        for vid in vids:
            url = f"https://www.youtube.com/watch?v={vid}"
            urls.append(url)
            print(url)
