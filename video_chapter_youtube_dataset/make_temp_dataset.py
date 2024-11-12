import json
import numpy as np

# Load the JSON data from file
with open('dataset/sampled_videos.json', 'r') as file:
    data = json.load(file)

# Initialize dictionaries to store training and validation IDs
train_data = {}
validation_data = {}
debugging_train_ids = []
debugging_val_ids = []

# Split the data for regular training and validation
for category, videos in data.items():
    np.random.shuffle(videos)  # Randomize the order of videos
    n_videos = len(videos)
    n_train = int(0.3 * n_videos)  # 30% for training
    n_validation = int(0.1 * n_videos)  # 10% for validation

    # Ensure that training and validation data do not overlap
    train_data[category] = videos[:n_train]
    validation_data[category] = videos[n_train:n_train + n_validation]

# Concatenate all videos into a single list for debugging purposes
all_videos = sum(data.values(), [])

# Randomly sample 100 IDs for debugging_train and 50 IDs for debugging_val
np.random.shuffle(all_videos)
debugging_train_ids = all_videos[:100]
debugging_val_ids = all_videos[100:150]

# Save the training and validation IDs to text files
with open('performance_train.txt', 'w') as file:
    for ids in train_data.values():
        file.write('\n'.join(ids) + '\n')

with open('performance_validation.txt', 'w') as file:
    for ids in validation_data.values():
        file.write('\n'.join(ids) + '\n')

# Save the debugging IDs to text files
with open('debugging_train.txt', 'w') as file:
    file.write('\n'.join(debugging_train_ids) + '\n')

with open('debugging_val.txt', 'w') as file:
    file.write('\n'.join(debugging_val_ids) + '\n')

print("All data splits have been saved to files successfully.")
