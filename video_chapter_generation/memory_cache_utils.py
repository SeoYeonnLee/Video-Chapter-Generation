import gc
import psutil
import logging
import threading
import time
from typing import Dict, Optional, Any, Callable
from collections import OrderedDict
import torch
import weakref

logger = logging.getLogger(__name__)

class SystemMemoryTracker:
    """대용량 메모리 환경(128GB RAM, 16GB 공유메모리)에 최적화된 메모리 추적 시스템"""
    
    def __init__(self):
        # 시스템 메모리 설정
        self.total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        self.shared_memory = 16 * 1024  # 16GB in MB
        self.available_memory = self.total_memory - self.shared_memory
        
        # 메모리 임계값 설정
        self.critical_threshold = 0.85 * self.available_memory
        self.warning_threshold = 0.75 * self.available_memory
        self.cleanup_threshold = 0.70 * self.available_memory
        
        # 모니터링 데이터 구조
        self.memory_history = []
        self.peak_memory = 0
        self.start_time = time.time()

        # 상태 관리 플래그
        self._is_shutdown = False
        self._tracking = False
        self._track_thread = None
        
        # 초기 설정 로깅
        logger.info(f"Memory Monitor initialized:")
        logger.info(f"Total System Memory: {self.total_memory/1024:.1f}GB")
        logger.info(f"Shared Memory: {self.shared_memory/1024:.1f}GB")
        logger.info(f"Available Memory: {self.available_memory/1024:.1f}GB")

    def start_tracking(self, interval: float = 5.0):
        """메모리 모니터링 시작"""
        self._tracking = True
        self._track_thread = threading.Thread(target=self._monitor_memory, args=(interval,))
        self._track_thread.daemon = True
        self._track_thread.start()

    def _monitor_memory(self, interval: float):
        """백그라운드 메모리 모니터링"""
        while self._tracking:
            current_memory = self._get_memory_status()
            self.peak_memory = max(self.peak_memory, current_memory['ram_used'])
            
            self.memory_history.append({
                'timestamp': time.time() - self.start_time,
                'memory': current_memory
            })
            
            if current_memory['ram_used'] > self.critical_threshold:
                logger.warning(f"Critical memory usage: {current_memory['ram_used']/1024:.1f}GB")
                
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
        """메모리 정리가 필요한지 확인"""
        if self._is_shutdown:
            return False
            
        status = self._get_memory_status()
        if status is None:
            return False
            
        return (status['ram_used'] > self.cleanup_threshold or 
                status['shared_memory_used'] > self.shared_memory * 0.8)

    def get_formatted_stats(self) -> str:
        """현재 메모리 상태를 포맷팅된 문자열로 반환"""
        status = self._get_memory_status()
        return (f"RAM: {status['ram_used']/1024:.1f}GB "
                f"Shared: {status['shared_memory_used']/1024:.1f}GB")

    def shutdown(self):
        """트래커 종료"""
        if not self._is_shutdown:
            self._tracking = False
            if self._track_thread:
                self._track_thread.join(timeout=1.0)
            self._is_shutdown = True

class CacheManager:
    """캐시 관리 시스템"""
    
    def __init__(self, max_cache_entries: int = 1000):
        self.max_cache_entries = max_cache_entries
        self.caches = weakref.WeakKeyDictionary()  # 객체별 캐시 관리
        
    def get_or_compute(self, 
                      owner: Any, 
                      cache_name: str, 
                      key: Any, 
                      compute_fn: Callable, 
                      *args, 
                      **kwargs) -> Any:
        """
        캐시된 값을 반환하거나 계산하여 캐시에 저장
        
        Args:
            owner: 캐시 소유 객체
            cache_name: 캐시 이름
            key: 캐시 키
            compute_fn: 값을 계산하는 함수
            *args, **kwargs: compute_fn에 전달할 인자들
        """
        if owner not in self.caches:
            self.caches[owner] = {}
        
        if cache_name not in self.caches[owner]:
            self.caches[owner][cache_name] = OrderedDict()
        
        cache = self.caches[owner][cache_name]
        
        if key in cache:
            value = cache.pop(key)
            cache[key] = value  # 최신 사용으로 갱신
            return value
            
        # 캐시 크기 제한 체크
        if len(cache) >= self.max_cache_entries:
            cache.popitem(last=False)
            
        # 새로운 값 계산 및 캐시
        value = compute_fn(*args, **kwargs)
        cache[key] = value
        return value
        
    def clear_cache(self, owner: Any = None, cache_name: str = None):
        """특정 캐시 또는 모든 캐시 초기화"""
        if owner is None:
            self.caches.clear()
        elif cache_name is None:
            if owner in self.caches:
                self.caches[owner].clear()
        else:
            if owner in self.caches and cache_name in self.caches[owner]:
                self.caches[owner][cache_name].clear()

class MemoryManager:
    """메모리 관리 통합 시스템"""
    
    def __init__(self):
        self.tracker = SystemMemoryTracker()
        self.cache_manager = CacheManager()
        self._is_shutdown = False
        self.tracker.start_tracking()
        
        # GPU 메모리 임계값
        self.gpu_warning_threshold = 0.75  # 75%
        self.gpu_critical_threshold = 0.85  # 85%
        
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
        """GPU 메모리 정리가 필요한지 확인"""
        ratio = self.check_gpu_memory()
        return ratio is not None and ratio > self.gpu_warning_threshold

    def cleanup(self, force: bool = False):
        """메모리 정리 수행"""
        if self._is_shutdown:
            return
            
        try:
            should_cleanup = force or self.tracker.should_cleanup() or self.should_cleanup_gpu()
            
            if should_cleanup:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # 캐시 정리
                if force:
                    self.cache_manager.clear_cache()
                
                self._last_cleanup_time = time.time()
                logger.info("Memory cleanup performed")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    def cleanup_dataloader(self, loader):
        """DataLoader 리소스 정리"""
        if self._is_shutdown:
            return
            
        try:
            if hasattr(loader, '_iterator'):
                del loader._iterator
            if hasattr(loader, '_workers'):
                for worker in loader._workers:
                    worker.terminate()
                loader._workers.clear()
            self.cleanup(force=True)
        except Exception as e:
            logger.warning(f"Error during dataloader cleanup: {str(e)}")

    def get_cache_manager(self) -> CacheManager:
        """캐시 매니저 인스턴스 반환"""
        return self.cache_manager

    def log_memory_stats(self, prefix: str = ""):
        """메모리 상태 로깅"""
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
                        
        except Exception as e:
            logger.warning(f"Error during memory stats logging: {str(e)}")

    def get_status_for_pbar(self) -> str:
        """프로그레스 바용 메모리 상태 문자열 반환"""
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
        """OOM 상황 처리"""
        self.cache_manager.clear_cache()  # 모든 캐시 초기화
        self.cleanup(force=True)
    
    def shutdown(self):
        """메모리 관리자 종료"""
        if not self._is_shutdown:
            self.cleanup(force=True)
            self.tracker.shutdown()
            self._is_shutdown = True

    def __del__(self):
        """소멸자"""
        try:
            self.shutdown()
        except Exception:
            pass