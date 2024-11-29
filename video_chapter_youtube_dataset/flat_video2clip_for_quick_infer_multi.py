"""
flat testing videos to clips, so that we can quickly test model performance without reloading dataloader many times

"""

import json
import os
import glob
import tempfile
import shutil
from tqdm import tqdm
from typing import List, Dict, Any
from dataset_process_scripts.load_dataset_utils import parse_csv_to_list, extract_first_timestamp

class JSONValidator:
    @staticmethod
    def validate_clip_info(clip_info: Dict) -> bool:
        """Validate structure of a single clip info"""
        required_fields = {
            "image_paths": list,
            "text_clip": str,
            "clip_label": int,
            "clip_start_end": list,
            "cut_points": list,
            "vid": str
        }
        
        for field, field_type in required_fields.items():
            if field not in clip_info:
                print(f"Missing required field: {field}")
                return False
            if not isinstance(clip_info[field], field_type):
                print(f"Invalid type for field {field}: expected {field_type}, got {type(clip_info[field])}")
                return False
        
        return True

def calculate_json_size(data: List[Dict]) -> int:
    """
    Estimate the size of JSON data in bytes
    """
    sample_size = min(100, len(data))
    if sample_size == 0:
        return 0
    
    sample = data[:sample_size]
    sample_json = json.dumps(sample, indent=2, ensure_ascii=False)
    avg_item_size = len(sample_json.encode('utf-8')) / sample_size
    
    estimated_size = avg_item_size * len(data)
    return int(estimated_size)

def save_json_safely(data: List[Dict], filepath: str) -> bool:
    """
    Safely save JSON data with validation and backup
    
    Args:
        data: List of clip information dictionaries
        filepath: Path to save the JSON file
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Validate data before saving
    if not all(JSONValidator.validate_clip_info(clip) for clip in data):
        print("Data validation failed")
        return False
    
    # Create temporary file
    temp_dir = os.path.dirname(filepath)
    fd, temp_path = tempfile.mkstemp(dir=temp_dir, suffix='.json', text=True)
    
    try:
        # Write to temporary file
        with os.fdopen(fd, 'w') as temp_file:
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        # Create backup if original file exists
        if os.path.exists(filepath):
            backup_path = filepath + '.backup'
            shutil.copy2(filepath, backup_path)
        
        # Atomic move of temporary file to final location
        shutil.move(temp_path, filepath)
        
        # Verify the saved file
        with open(filepath, 'r') as f:
            verify_data = json.load(f)
            if len(verify_data) != len(data):
                raise ValueError("Verification failed: data length mismatch")
        
        # Remove backup if everything succeeded
        if os.path.exists(filepath + '.backup'):
            os.remove(filepath + '.backup')
            
        print(f"Successfully saved {len(data)} clips to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
        # Restore from backup if it exists
        backup_path = filepath + '.backup'
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, filepath)
            print("Restored from backup")
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def save_json_in_chunks(data: List[Dict], base_filepath: str, target_chunk_size_gb: float = 1.0) -> bool:
    """
    Save data in multiple JSON files, each approximately target_chunk_size_gb in size
    """
    if not data:
        print("No data to save")
        return False
        
    target_chunk_size = int(target_chunk_size_gb * 1024 * 1024 * 1024)
    
    total_estimated_size = calculate_json_size(data)
    items_per_chunk = int(len(data) * target_chunk_size / total_estimated_size)
    items_per_chunk = max(1, items_per_chunk)
    
    print(f"Estimated total size: {total_estimated_size / (1024*1024*1024):.2f} GB")
    print(f"Items per chunk: {items_per_chunk}")
    
    chunks = []
    for i in range(0, len(data), items_per_chunk):
        chunks.append(data[i:i + items_per_chunk])
    
    print(f"Splitting into {len(chunks)} chunks")
    
    all_success = True
    for i, chunk in enumerate(chunks):
        chunk_filepath = f"{base_filepath}_{i+1}.json"
        print(f"Saving chunk {i+1}/{len(chunks)} to {chunk_filepath}")
        
        if not save_json_safely(chunk, chunk_filepath):
            print(f"Failed to save chunk {i+1}")
            all_success = False
            break
            
        chunk_size = os.path.getsize(chunk_filepath) / (1024 * 1024 * 1024)
        print(f"Chunk {i+1} size: {chunk_size:.2f} GB")
    
    if all_success:
        metadata = {
            "total_items": len(data),
            "chunk_count": len(chunks),
            "items_per_chunk": items_per_chunk,
            "chunk_files": [f"part_{i+1}.json" for i in range(len(chunks))]
        }
        
        metadata_filepath = f"{base_filepath}_metadata.json"
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Successfully saved all {len(chunks)} chunks")
        print(f"Metadata saved to {metadata_filepath}")
    
    return all_success

def flat_videos2clips(img_dir: str, data_file: str, vid_files: List[str], clip_frame_num: int = 16) -> List[Dict]:
    """Flat videos to clips with progress tracking and validation"""
    half_clip_frame_num = int(clip_frame_num // 2)
    all_clip_infos = []
    
    for vid_file in vid_files:
        print(f"Processing vid_file: {vid_file}")
        
        try:
            # Load video IDs
            with open(vid_file, "r") as f:
                vids = [x.strip() for x in f.readlines()]
            
            # Load basic data
            all_vids, titles, durations, timestamps = parse_csv_to_list(data_file)
            
            # Load subtitle files
            asr_file_list = glob.glob(os.path.dirname(data_file) + "/*/subtitle_*.json")
            asr_files = {os.path.basename(asr_file).split(".")[0][9:]: asr_file for asr_file in asr_file_list}
            
            no_keys = [(i, item) for i, item in enumerate(vids) if item not in asr_files]
            if no_keys:
                print(f"Warning: No subtitles for these vids: {no_keys}")
            
            total_vids = len(vids)
            for idx, vid in enumerate(tqdm(vids, desc=f"Processing videos from {vid_file}", position=0), 1):                
                try:
                    i = all_vids.index(vid)
                    title, duration, timestamp = titles[i], durations[i], timestamps[i]
                    
                    # Load subtitles
                    with open(asr_files[vid], "r") as f:
                        subtitles = json.load(f)
                    
                    # Count images
                    image_path = os.path.join(img_dir, vid)
                    image_num = len(glob.glob(image_path + "/*.jpg"))
                    
                    if image_num == 0:
                        print(f"Warning: No images found for vid {vid}")
                        continue
                    
                    # Process cut points
                    cut_points, descriptions = [], []
                    for timestamp_str in timestamp:
                        sec, description = extract_first_timestamp(timestamp_str)
                        if 4 <= sec <= image_num - 4:
                            cut_points.append(sec)
                            descriptions.append(description)
                    
                    # Process clips
                    max_offset = 2
                    clips = [[start_t, start_t + clip_frame_num] 
                            for start_t in range(0, image_num - clip_frame_num, 8 * max_offset)]
                    
                    for clip_start_sec, clip_end_sec in clips:
                        clip_info = process_single_clip(
                            vid, clip_start_sec, clip_end_sec, 
                            cut_points, half_clip_frame_num,
                            subtitles, image_path, image_num,
                            clip_frame_num, max_offset
                        )
                        
                        if clip_info and JSONValidator.validate_clip_info(clip_info):
                            all_clip_infos.append(clip_info)
                        
                except Exception as e:
                    print(f"Error processing video {vid}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Error processing file {vid_file}: {str(e)}")
            continue
    
    return all_clip_infos

def process_single_clip(vid, clip_start_sec, clip_end_sec, cut_points, 
                       half_clip_frame_num, subtitles, image_path, image_num,
                       clip_frame_num, max_offset) -> Dict[str, Any]:
    """Process a single clip and return clip information"""
    # Calculate label
    label = 0
    for cp in cut_points:
        pos_st = cp - half_clip_frame_num
        pos_et = cp + half_clip_frame_num
        a = max(clip_start_sec, pos_st)
        mi = min(clip_start_sec, pos_st)
        b = min(clip_end_sec, pos_et)
        ma = max(clip_end_sec, pos_et)
        
        iou = (b - a) / (ma - mi)
        if iou >= (clip_frame_num - max_offset) / (clip_frame_num + max_offset):
            label = 1
    
    # Get subtitle text
    text_extra_time_gap = 1
    text_clip = ""
    for sub in subtitles:
        if clip_start_sec - text_extra_time_gap < sub["start"] < clip_end_sec + text_extra_time_gap:
            text_clip = text_clip + " " + sub["text"] if text_clip else sub["text"]
    
    # Get image paths
    img_path_list = []
    for idx in range(clip_start_sec, clip_end_sec):
        if clip_start_sec <= 2 or clip_start_sec >= image_num - clip_frame_num - 2:
            image_filename = f"{idx+1:05d}.jpg"
        else:
            image_filename = f"{idx+3:05d}.jpg"
        
        image_filename = os.path.join(image_path, image_filename)
        if not os.path.exists(image_filename):
            print(f"Warning: Image file not found: {image_filename}")
            continue
        img_path_list.append(image_filename)
    
    return {
        "image_paths": img_path_list,
        "text_clip": text_clip,
        "clip_label": label,
        "clip_start_end": [clip_start_sec, clip_end_sec],
        "cut_points": cut_points,
        "vid": vid
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='video chapter model')
    parser.add_argument('--clip_frame_num', default=16, type=int)
    parser.add_argument('--chunk_size_gb', default=1.0, type=float)
    args = parser.parse_args()

    img_dir = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/youtube_video_frame_dataset"
    data_file = "/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/all_in_one_with_subtitle_final.csv"
    vid_files = ["dataset/final_validation.txt"]
    # vid_files = ["dataset/final_train.txt", "dataset/final_validation.txt"]

    clip_frame_num = args.clip_frame_num
    all_clip_infos = flat_videos2clips(img_dir, data_file, vid_files, clip_frame_num)

    base_save_path = f"/home/work/capstone/Video-Chapter-Generation/video_chapter_youtube_dataset/dataset/validation_clips_clip_frame_num_offset8_{clip_frame_num}"

    if save_json_in_chunks(all_clip_infos, base_save_path, args.chunk_size_gb):
        print("Successfully completed processing and saving")
    else:
        print("Error occurred during save process")