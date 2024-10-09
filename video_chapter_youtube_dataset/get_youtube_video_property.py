"""
Get video chapter from youtube
https://stackoverflow.com/questions/63821605/how-do-i-get-info-about-a-youtube-videos-chapters-from-the-api

Need to get a access key from google console
https://console.cloud.google.com/apis/credentials/wizard?folder=&organizationId=&project=glossy-receiver-315011

"""


import requests
import re


def parse_timestamp(description):
    timestamp_lines = []

    lines = description.split("\n")
    for i, line in enumerate(lines):
        if len(line) > 150:
            continue

        if len(timestamp_lines) == 0 and "0:00" in line:
            line = re.sub(r'http\S+', '', line)  # remove all http urls
            timestamp_lines.append(line)
            continue

        if len(timestamp_lines) > 0:
            if re.search("\d{1}:\d{2}", line):
                line = re.sub(r'http\S+', '', line)     # remove all http urls
                timestamp_lines.append(line)

    return timestamp_lines


URL = "https://www.googleapis.com/youtube/v3/videos?part=snippet&id=-4rrfrtTX84&key=AIzaSyB-wUr9YQt8LOwlq7B19oJr7ViFI5VA-L0"

r = requests.get(url=URL)
data = r.json()
description = data["items"][0]["snippet"]["description"]
timestamp_lines = parse_timestamp(description)

print()
