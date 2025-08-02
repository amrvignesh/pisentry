# resource_monitor.py
# Comprehensive resource monitoring module for PiSentry
# Tracks CPU, memory, inference time, and energy consumption (if hardware available)

import psutil
import time
import threading
import json
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class PiSentryResourceMonitor:
    """
    Comprehensive resource monitoring for PiSentry deployment.
    Tracks performance metrics critical for IoT device deployment.
    """
    
    def __init__(self, max_history=1000):
        self.process = psutil.Process()
        self.max_history = max_history
        self.baseline_memory = None
        self.baseline_cpu = None
        self.monitoring_active = False
        
        # Performance history storage
        self.metrics_history = {
            'timestamps': deque(maxlen=max_history),
            'memory_usage_mb': deque(maxlen=max_history),
            'cpu_percent': deque(maxlen=max_history),
            'inference_times_ms': deque(maxlen=max_history),
            'prediction_results': deque(maxlen=max_history),
            'system_load': deque(maxlen=max_history)
        }
        
        # Performance targets from paper
        self.targets = {
            'max_memory_mb': 10,
            'max_cpu_percent': 20,
            'max_inference_ms': 250,
            'min_f1_score': 0.92
        }
        
        # Establish baseline
        self._establish_baseline()
    
    def _establish_baseline(self):
        """Establish baseline resource usage before ML operations."""
        print("Establishing resource usage baseline...")
        
        # Take multiple measurements for accurate baseline
        memory_readings = []
        cpu_readings = []
        
        for _ in range(10):
            memory_readings.append(self.process.memory_info().rss / 1024 / 1024)
            cpu_readings.append(self.process.cpu_percent(interval=0.1))
            time.sleep(0.1)
        
        self.baseline_memory = np.mean(memory_readings)
        self.baseline_cpu = np.mean(cpu_readings)
        
        print(f"Baseline established - Memory: {self.baseline_memory:.1f} MB, CPU: {self.baseline_cpu:.1f}%")
    
    @contextmanager
    def monitor_inference(self):
        """
        Context manager for monitoring a single inference operation.
        Returns detailed metrics for the inference.
        """
        # Pre-inference measurements
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # System load
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        
        try:
            yield
        finally:
            # Post-inference measurements
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            # Calculate metrics
            inference_time_ms = (end_time - start_time) * 1000
            memory_increase_mb = end_memory - start_memory
            
            # Store measurements
            timestamp = datetime.now()
            self.metrics_history['timestamps'].append(timestamp)
            self.metrics_history['memory_usage_mb'].append(end_memory)
            self.metrics_history['cpu_percent'].append(cpu_percent)
            self.metrics_history['inference_times_ms'].append(inference_time_ms)
            self.metrics_history['system_load'].append(load_avg)
            
            # Return metrics
            metrics = {
                'timestamp': timestamp.isoformat(),
                'inference_time_ms': inference_time_ms,
                'memory_usage_mb': end_memory,
                'memory_increase_mb': memory_increase_mb,
                'cpu_percent': cpu_percent,
                'system_load': load_avg,
                'meets_targets': {
                    'memory': end_memory <= self.targets['max_memory_mb'],
                    'cpu': cpu_percent <= self.targets['max_cpu_percent'],
                    'inference_time': inference_time_ms <= self.targets['max_inference_ms']
                }
            }
            
            # Log warnings if targets exceeded
            if not metrics['meets_targets']['memory']:
                logger.warning(f"Memory usage {end_memory:.1f} MB exceeds target {self.targets['max_memory_mb']} MB")
            if not metrics['meets_targets']['cpu']:
                logger.warning(f"CPU usage {cpu_percent:.1f}% exceeds target {self.targets['max_cpu_percent']}%")
            if not metrics['meets_targets']['inference_time']:
                logger.warning(f"Inference time {inference_time_ms:.2f} ms exceeds target {self.targets['max_inference_ms']} ms")
    
    def start_continuous_monitoring(self, interval=1.0):
        """
        Start continuous background monitoring of system resources.
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Collect current metrics
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    cpu_percent = self.process.cpu_percent()
                    load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
                    
                    # Store in history
                    timestamp = datetime.now()
                    self.metrics_history['timestamps'].append(timestamp)
                    self.metrics_history['memory_usage_mb'].append(memory_mb)
                    self.metrics_history['cpu_percent'].append(cpu_percent)
                    self.metrics_history['system_load'].append(load_avg)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(interval)
        
        # Start monitoring in background thread
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started continuous resource monitoring (interval: {interval}s)")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped continuous resource monitoring")
    
    def get_performance_summary(self):
        """
        Generate comprehensive performance summary.
        """
        if not self.metrics_history['memory_usage_mb']:
            return "No performance data available"
        
        # Calculate statistics
        memory_stats = {
            'mean': np.mean(self.metrics_history['memory_usage_mb']),
            'max': np.max(self.metrics_history['memory_usage_mb']),
            'min': np.min(self.metrics_history['memory_usage_mb']),
            'std': np.std(self.metrics_history['memory_usage_mb'])
        }
        
        cpu_stats = {
            'mean': np.mean(self.metrics_history['cpu_percent']),
            'max': np.max(self.metrics_history['cpu_percent']),
            'min': np.min(self.metrics_history['cpu_percent']),
            'std': np.std(self.metrics_history['cpu_percent'])
        }
        
        inference_stats = {}
        if self.metrics_history['inference_times_ms']:
            inference_stats = {
                'mean': np.mean(self.metrics_history['inference_times_ms']),
                'max': np.max(self.metrics_history['inference_times_ms']),
                'min': np.min(self.metrics_history['inference_times_ms']),
                'p95': np.percentile(self.metrics_history['inference_times_ms'], 95),
                'p99': np.percentile(self.metrics_history['inference_times_ms'], 99)
            }
        
        # Check target compliance
        memory_compliant = memory_stats['mean'] <= self.targets['max_memory
