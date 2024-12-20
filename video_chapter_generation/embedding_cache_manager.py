# embedding_cache_manager.py

import os
import torch
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import h5py
from typing import Dict, List, Tuple

class EmbeddingCacheManager:
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._video_embeddings: Dict[str, np.memmap] = {}
        
    @lru_cache(maxsize=1000)
    def _load_single_embedding(self, embedding_file: str) -> np.ndarray:
        return np.load(embedding_file)
    
    def load_video_embeddings(self, video_id: str, embedding_dir: str) -> np.memmap:
        """Load video embeddings using memory mapping"""
        if video_id not in self._video_embeddings:
            video_path = os.path.join(embedding_dir, video_id)
            embedding_files = sorted(glob.glob(os.path.join(video_path, "vision_emb_*.npy")))
            if not embedding_files:
                raise FileNotFoundError(f"No embedding files found for video {video_id}")
                
            # Create memory mapped array
            sample_embedding = np.load(embedding_files[0])
            shape = (len(embedding_files), *sample_embedding.shape)
            memmap_path = os.path.join(embedding_dir, f"{video_id}_memmap.npy")
            
            if not os.path.exists(memmap_path):
                # Create new memmap file
                memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=shape)
                for i, f in enumerate(embedding_files):
                    memmap[i] = np.load(f)
                memmap.flush()
            
            # Load existing memmap
            self._video_embeddings[video_id] = np.memmap(
                memmap_path, dtype='float32', mode='r', shape=shape
            )
            
        return self._video_embeddings[video_id]
    
    def parallel_load_embeddings(
        self, 
        video_ids: List[str], 
        clip_start_frames: List[List[int]], 
        embedding_dir: str
    ) -> torch.Tensor:
        """Load embeddings in parallel"""
        futures = []
        
        for video_id, start_frames in zip(video_ids, clip_start_frames):
            future = self.thread_pool.submit(
                self._load_video_clip_embeddings,
                video_id,
                start_frames,
                embedding_dir
            )
            futures.append(future)
        
        results = [future.result() for future in futures]
        return torch.stack([torch.from_numpy(r) for r in results])
    
    def _load_video_clip_embeddings(
        self,
        video_id: str,
        start_frames: List[int],
        embedding_dir: str
    ) -> np.ndarray:
        """Load embeddings for specific clips of a video"""
        video_embeddings = self.load_video_embeddings(video_id, embedding_dir)
        clip_embeddings = []
        
        for start_frame in start_frames:
            if start_frame == -1:
                # Padding case
                padding_emb = np.zeros((16, 2048), dtype=np.float32)
                clip_embeddings.append(padding_emb)
            else:
                # Load from memmap
                clip_embeddings.append(video_embeddings[start_frame:start_frame+16])
                
        return np.stack(clip_embeddings)
    
    def create_merged_embedding_file(
        self,
        video_ids: List[str], 
        embedding_dir: str,
        output_file: str
    ) -> None:
        """Merge all embeddings into a single H5 file"""
        with h5py.File(output_file, 'w') as f:
            for video_id in video_ids:
                embeddings = self.load_video_embeddings(video_id, embedding_dir)
                f.create_dataset(video_id, data=embeddings)
    
    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown()
        self._video_embeddings.clear()
        self._load_single_embedding.cache_clear()