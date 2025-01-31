import gc
import psutil
import logging
import threading
import time
import queue
from typing import Dict, Optional, Any, Callable
from collections import OrderedDict
import torch

logger = logging.getLogger(__name__)

class CacheEntry:
    def __init__(self, value, timestamp=None):
        self.value = value
        self.timestamp = timestamp or time.time()
        self.access_count = 0

    def access(self):
        self.access_count += 1
        self.timestamp = time.time()

class AsyncCacheManager:    
    def __init__(self, max_cache_entries: int = 1000, cleanup_threshold: float = 0.8):
        self.max_cache_entries = max_cache_entries
        self.cleanup_threshold = cleanup_threshold
        self.caches = {}
        self.cache_hits = 0
        self.cache_misses = 0

        self.cache_queue = queue.Queue()
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_cache_queue, daemon=True)
        self.worker_thread.start()

        self.last_cleanup = time.time()
        self.cleanup_interval = 3600 # 1 hour
    
    def _process_cache_queue(self):
        """비동기적으로 캐시 업데이트를 처리하는 워커 스레드"""
        while self.is_running:
            try:
                task = self.cache_queue.get(timeout=1.0)
                if task is None:  # 종료 신호
                    break
                    
                owner, cache_name, key, value = task
                self._update_cache(owner, cache_name, key, value)
                self.cache_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in cache worker: {str(e)}")

    def _update_cache(self, owner, cache_name, key, value):
        """캐시 업데이트 로직"""
        if owner not in self.caches:
            self.caches[owner] = {}
        
        if cache_name not in self.caches[owner]:
            self.caches[owner][cache_name] = OrderedDict()
        
        cache = self.caches[owner][cache_name]
        
        if len(cache) >= self.max_cache_entries:
            # 가장 오래된 항목 제거
            cache.popitem(last=False)
        
        cache[key] = CacheEntry(value)
    
    def get_or_compute(self, owner, cache_name, key, compute_fn, *args, **kwargs):
        # 학습 중일 때는 항상 새로 계산
        if owner.training:
            return compute_fn(*args, **kwargs)
            
        # 추론 시에만 캐시 사용
        if owner not in self.caches:
            self.caches[owner] = {}
        
        if cache_name not in self.caches[owner]:
            self.caches[owner][cache_name] = OrderedDict()
        
        cache = self.caches[owner][cache_name]
        
        if key in cache:
            self.cache_hits += 1
            entry = cache[key]
            entry.access()
            return entry.value
            
        self.cache_misses += 1
        value = compute_fn(*args, **kwargs)
        self.cache_queue.put((owner, cache_name, key, value))
        
        return value
    
    def async_update(self, owner, cache_name, key, value):
        """비동기적으로 캐시 업데이트 요청"""
        self.cache_queue.put((owner, cache_name, key, value))

    def clear_cache(self, owner: Any = None, cache_name: str = None):
        """Clear specific or all caches"""
        if owner is None:
            self.caches.clear()
        elif cache_name is None and owner in self.caches:
            self.caches[owner].clear()
        elif owner in self.caches and cache_name in self.caches[owner]:
            self.caches[owner][cache_name].clear()

    def cleanup_old_entries(self):
        """Clean up old cache entries based on access patterns"""
        current_time = time.time()
        
        for owner in self.caches:
            for cache_name in self.caches[owner]:
                cache = self.caches[owner][cache_name]
                
                # 접근 빈도와 최근 사용 시간 기반 정리
                entries = sorted(
                    cache.items(),
                    key=lambda x: (
                        x[1].access_count,  # 접근 빈도
                        -(current_time - x[1].timestamp)  # 최근 사용 시간
                    )
                )
                
                # 하위 20%만 제거하여 점진적 정리
                num_to_remove = len(entries) // 5
                for key, _ in entries[:num_to_remove]:
                    del cache[key]
    

    def get_stats(self):
        """Get cache performance statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_ratio': hit_ratio,
            'total_entries': sum(
                len(cache)
                for owner_caches in self.caches.values()
                for cache in owner_caches.values()
            )
        }
    
    def shutdown(self):
        """캐시 매니저 종료"""
        self.is_running = False
        self.cache_queue.put(None)  # 워커 스레드에 종료 신호 전송
        self.worker_thread.join()

class SystemMemoryTracker:
    def __init__(self, warning_threshold: float = 0.75, critical_threshold: float = 0.85):
        # 시스템 메모리 설정
        self.total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        self.shared_memory = 16 * 1024  # 16GB in MB
        self.available_memory = self.total_memory - self.shared_memory
        
        # 메모리 임계값 설정
        self.critical_threshold = critical_threshold * self.available_memory
        self.warning_threshold = warning_threshold * self.available_memory
        self.cleanup_threshold = (warning_threshold - 0.1) * self.available_memory
        
        # 모니터링 데이터 구조
        self.memory_history = []
        self.peak_memory = 0
        self.start_time = time.time()
        self._is_shutdown = False
        self._tracking = False
        self._track_thread = None
        
        logger.info(f"Memory Monitor initialized:")
        logger.info(f"Total System Memory: {self.total_memory/1024:.1f}GB")
        logger.info(f"Shared Memory: {self.shared_memory/1024:.1f}GB")
        logger.info(f"Available Memory: {self.available_memory/1024:.1f}GB")

    def start_tracking(self, interval: float = 5.0):
        self._tracking = True
        self._track_thread = threading.Thread(target=self._monitor_memory, args=(interval,))
        self._track_thread.daemon = True
        self._track_thread.start()

    def _monitor_memory(self, interval: float):
        """백그라운드 메모리 모니터링"""
        while self._tracking:
            status = self._get_memory_status()
            if status:
                self.peak_memory = max(self.peak_memory, status['ram_used'])
            
                self.memory_history.append({
                    'timestamp': time.time() - self.start_time,
                    'memory': status
                })
                
                if status['ram_used'] > self.critical_threshold:
                    logger.warning(f"Critical memory usage: {status['ram_used']/1024:.1f}GB")
                
            time.sleep(interval)

    def _get_memory_status(self) -> Optional[Dict[str, float]]:
        """현재 메모리 상태를 확인"""
        if self._is_shutdown:
            return None
            
        try:
            process = psutil.Process()
            virtual_memory = psutil.virtual_memory()
            
            return {
                'ram_used': process.memory_info().rss / (1024 * 1024),
                'ram_percent': (process.memory_info().rss / self.available_memory) * 100,
                'system_used': (virtual_memory.total - virtual_memory.available) / (1024 * 1024),
                'system_percent': virtual_memory.percent,
                'shared_memory_used': process.memory_info().shared / (1024 * 1024)
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            return None

    def should_cleanup(self) -> bool:        
        status = self._get_memory_status()
        if status is None:
            return False
            
        return (status['ram_used'] > self.cleanup_threshold or 
                status['shared_memory_used'] > self.shared_memory * 0.8)

    def get_formatted_stats(self) -> str:
        """현재 메모리 상태를 포맷팅된 문자열로 반환"""
        status = self._get_memory_status()
        if status:
            return (f"RAM: {status['ram_used']/1024:.1f}GB "
                    f"Shared: {status['shared_memory_used']/1024:.1f}GB")
        return "Memory status unavailable"

    def shutdown(self):
        """트래커 종료"""
        if not self._is_shutdown:
            self._tracking = False
            if self._track_thread:
                self._track_thread.join(timeout=1.0)
            self._is_shutdown = True


class MemoryManager:    
    def __init__(self, warning_threshold: float = 0.75, critical_threshold: float = 0.85):
        self.tracker = SystemMemoryTracker(warning_threshold, critical_threshold)
        self.cache_manager = AsyncCacheManager()
        self._is_shutdown = False
        self.tracker.start_tracking()
        
        # GPU 메모리 임계값
        self.gpu_warning_threshold = warning_threshold
        self.gpu_critical_threshold = critical_threshold
        
        # 캐시 설정
        self.max_cache_size = int(self.tracker.available_memory * 0.3)  # 30% of available memory
        self._cache_usage = 0
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 600  # 10분
        
        logger.info("MemoryManager initialized")

    def check_gpu_memory(self) -> Optional[float]:
        """GPU 메모리 사용률 확인"""
        if not torch.cuda.is_available():
            return None
            
        current = torch.cuda.memory_allocated()
        maximum = torch.cuda.max_memory_allocated()
        
        if maximum == 0:
            return 0.0
            
        return current / maximum

    def should_cleanup_gpu(self) -> bool:
        ratio = self.check_gpu_memory()
        return ratio is not None and ratio > self.gpu_warning_threshold

    def cleanup(self, force: bool = False):
        if self._is_shutdown:
            return
            
        try:
            should_cleanup = force or self.tracker.should_cleanup()
            
            if should_cleanup:
                gc.collect()
                if torch and torch.cuda and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # 캐시 정리
                if force:
                    self.cache_manager.clear_cache()
                else:
                    self.cache_manager.cleanup_old_entries()
                
                self._last_cleanup_time = time.time()
                logger.info("Memory cleanup performed")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    def cleanup_if_needed(self, epoch, is_training):
        # 현재 메모리 상태 확인
        memory_status = self.tracker._get_memory_status()
        
        # 학습 중이 아니고, 메모리 사용량이 임계치를 넘을 때만 정리
        if not is_training and memory_status:
            if memory_status['ram_used'] > self.tracker.warning_threshold:
                # 점진적 정리
                self.cache_manager.cleanup_old_entries()
                if memory_status['ram_used'] > self.tracker.critical_threshold:
                    # 위험 수준일 때만 강제 정리
                    self.cleanup(force=True)
    
    def get_cache_manager(self) -> AsyncCacheManager:
        return self.cache_manager

    def log_memory_stats(self, prefix: str = ""):
        if self._is_shutdown:
            return
            
        try:
            status = self.tracker._get_memory_status()
            if status:
                logger.info(f"{prefix} Memory Status:")
                logger.info(f"RAM Usage: {status['ram_used']/1024:.1f}GB")
                logger.info(f"Shared Memory: {status['shared_memory_used']/1024:.1f}GB")
                logger.info(f"System Memory: {status['system_percent']:.1f}%")
                
                if torch.cuda.is_available():
                    gpu_ratio = self.check_gpu_memory()
                    if gpu_ratio is not None:
                        logger.info(f"GPU Memory Usage: {gpu_ratio*100:.1f}%")

                cache_stats = self.cache_manager.get_stats()
                logger.info(f"Cache Hit Ratio: {cache_stats['hit_ratio']*100:.1f}%")
                        
        except Exception as e:
            logger.warning(f"Error during memory stats logging: {str(e)}")

    def get_status_for_pbar(self) -> str:
        if self._is_shutdown:
            return "Memory tracking disabled"
            
        try:
            status = self.tracker._get_memory_status()
            if status:
                gpu_status = ""
                if torch.cuda.is_available():
                    gpu_ratio = self.check_gpu_memory()
                    if gpu_ratio is not None:
                        gpu_status = f" GPU: {gpu_ratio*100:.1f}%"
                return f"RAM: {status['ram_used']/1024:.1f}GB{gpu_status}"
        except Exception:
            pass
        return "Memory status unavailable"

    def handle_oom(self):
        try:
            self.cache_manager.clear_cache()
            self.cleanup(force=True)
        
        except Exception as e:
            logger.error(f"Error handling OOM: {str(e)}")

    def cleanup_dataloader(self, loader):
        if self._is_shutdown:
            return
            
        try:
            if hasattr(loader, '_iterator'):
                del loader._iterator
            if hasattr(loader, '_workers'):
                for worker in loader._workers:
                    worker.terminate()
                loader._workers.clear()
            self.cleanup(force=False)
        except Exception as e:
            logger.warning(f"Error during dataloader cleanup: {str(e)}")
    
    def shutdown(self):
        if not self._is_shutdown:
            self.cleanup(force=True)
            self.cache_manager.shutdown()
            self.tracker.shutdown()
            self._is_shutdown = True

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    