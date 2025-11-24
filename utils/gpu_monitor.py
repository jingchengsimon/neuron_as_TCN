import time
import numpy as np
import torch
import torch.cuda as cuda

# Add GPU monitoring library
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py")
    GPU_MONITORING_AVAILABLE = False

class GPUMonitor:
    """GPU monitoring class for monitoring GPU usage and performance"""
    
    def __init__(self, gpu_index=0):
        """
        Initialize GPU monitor
        
        Args:
            gpu_index: GPU device index, default is 0
        """
        self.gpu_index = gpu_index
        self.available = GPU_MONITORING_AVAILABLE
        self.handle = None
        
        if self.available:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                gpu_name_bytes = pynvml.nvmlDeviceGetName(self.handle)
                # Handle different versions of pynvml return format
                if isinstance(gpu_name_bytes, bytes):
                    self.gpu_name = gpu_name_bytes.decode('utf-8')
                else:
                    self.gpu_name = str(gpu_name_bytes)
                self.total_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total
            except Exception as e:
                print(f"Error initializing GPU monitor: {e}")
                self.available = False
    
    def get_utilization(self):
        """Get GPU utilization"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return utilization.gpu
        except:
            return "Error"
    
    def get_memory_usage(self):
        """Get GPU memory usage"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_mb = memory_info.used / 1024 / 1024
            total_mb = memory_info.total / 1024 / 1024
            return f"{used_mb:.0f}/{total_mb:.0f} MB"
        except:
            return "Error"
    
    def get_memory_percent(self):
        """Get GPU memory usage percentage"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return (memory_info.used / memory_info.total) * 100
        except:
            return "Error"
    
    def get_temperature(self):
        """Get GPU temperature"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except:
            return "Error"
    
    def get_power_usage(self):
        """Get GPU power usage"""
        if not self.available or not self.handle:
            return "N/A"
        
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
            return power
        except:
            return "Error"
    
    def get_comprehensive_info(self):
        """Get comprehensive GPU information"""
        if not self.available:
            return {
                'name': 'N/A',
                'utilization': 'N/A',
                'memory_usage': 'N/A',
                'memory_percent': 'N/A',
                'temperature': 'N/A',
                'power': 'N/A'
            }
        
        return {
            'name': self.gpu_name if hasattr(self, 'gpu_name') else 'N/A',
            'utilization': self.get_utilization(),
            'memory_usage': self.get_memory_usage(),
            'memory_percent': self.get_memory_percent(),
            'temperature': self.get_temperature(),
            'power': self.get_power_usage()
        }
    
    def print_status(self, prefix=""):
        """Print current GPU status"""
        info = self.get_comprehensive_info()
        print(f"{prefix}GPU: {info['name']}")
        print(f"{prefix}Utilization: {info['utilization']}%")
        print(f"{prefix}Memory: {info['memory_usage']} ({info['memory_percent']}%)")
        print(f"{prefix}Temperature: {info['temperature']}°C")
        print(f"{prefix}Power: {info['power']}W")
    
    def monitor_continuously(self, duration_seconds=300, interval_seconds=5, output_file=None):
        """
        Continuously monitor GPU usage
        
        Args:
            duration_seconds: Monitoring duration (seconds)
            interval_seconds: Monitoring interval (seconds)
            output_file: Output file path (optional)
        """
        if not self.available:
            print("GPU monitoring not available")
            return
        
        print(f"\nStarting GPU monitoring ({duration_seconds}s, recording every {interval_seconds}s):")
        print("Timestamp        | GPU Util(%) | Memory(MB) | Memory(%) | Temp(°C) | Power(W)")
        print("-" * 90)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write("timestamp,utilization,memory_used,memory_percent,temperature,power\n")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        try:
            while time.time() < end_time:
                elapsed = time.time() - start_time
                info = self.get_comprehensive_info()
                
                timestamp = time.strftime("%H:%M:%S")
                line = f"{timestamp} | {info['utilization']:11} | {info['memory_usage']:10} | {info['memory_percent']:12.1f} | {info['temperature']:8} | {info['power']:6}"
                print(line)
                
                if output_file:
                    with open(output_file, 'a') as f:
                        f.write(f"{timestamp},{info['utilization']},{info['memory_usage']},{info['memory_percent']},{info['temperature']},{info['power']}\n")
                
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
        
        print("-" * 90)
        print("GPU monitoring completed")
    
    def benchmark_performance(self, test_duration=60):
        """
        Performance benchmark test
        
        Args:
            test_duration: Test duration (seconds)
        """
        if not self.available:
            print("GPU monitoring not available for benchmarking")
            return
        
        print(f"\nStarting GPU performance benchmark test ({test_duration}s)...")
        
        # Collect performance data
        utilizations = []
        memory_percents = []
        temperatures = []
        power_usages = []
        
        start_time = time.time()
        end_time = start_time + test_duration
        
        while time.time() < end_time:
            info = self.get_comprehensive_info()
            
            if info['utilization'] != 'N/A' and info['utilization'] != 'Error':
                utilizations.append(info['utilization'])
            if info['memory_percent'] != 'N/A' and info['memory_percent'] != 'Error':
                memory_percents.append(info['memory_percent'])
            if info['temperature'] != 'N/A' and info['temperature'] != 'Error':
                temperatures.append(info['temperature'])
            if info['power'] != 'N/A' and info['power'] != 'Error':
                power_usages.append(info['power'])
            
            time.sleep(1)
        
        # Calculate statistics
        print("\n=== Performance Benchmark Results ===")
        if utilizations:
            print(f"GPU Utilization - Avg: {np.mean(utilizations):.1f}%, Max: {np.max(utilizations):.1f}%, Min: {np.min(utilizations):.1f}%")
        if memory_percents:
            print(f"Memory Usage - Avg: {np.mean(memory_percents):.1f}%, Max: {np.max(memory_percents):.1f}%, Min: {np.min(memory_percents):.1f}%")
        if temperatures:
            print(f"GPU Temperature - Avg: {np.mean(temperatures):.1f}°C, Max: {np.max(temperatures):.1f}°C, Min: {np.min(temperatures):.1f}°C")
        if power_usages:
            print(f"GPU Power - Avg: {np.mean(power_usages):.1f}W, Max: {np.max(power_usages):.1f}W, Min: {np.min(power_usages):.1f}W")
        
        # Performance evaluation
        avg_utilization = np.mean(utilizations) if utilizations else 0
        if avg_utilization > 80:
            print("✓ GPU utilization excellent (>80%)")
        elif avg_utilization > 50:
            print("○ GPU utilization good (50-80%)")
        elif avg_utilization > 20:
            print("⚠ GPU utilization low (20-50%)")
        else:
            print("✗ GPU utilization too low (<20%)")
        
        print("========================")

def configure_pytorch_gpu():
    """
    Configure PyTorch GPU settings
    """
    print("\n=== PyTorch GPU Configuration ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {cuda.device_count()}")
    
    if cuda.is_available():
        for i in range(cuda.device_count()):
            gpu_name = cuda.get_device_name(i)
            gpu_memory = cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set current device
        device = torch.device('cuda:0')
        cuda.set_device(0)
        
        # Enable cudnn benchmark for performance optimization
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocation strategy
        cuda.empty_cache()
        
        print(f"Current device: {device}")
        print("CUDNN benchmark: enabled")
        print("========================")
        
        return device
    else:
        print("CUDA not available, using CPU")
        print("========================")
        return torch.device('cpu')

def get_gpu_memory_info():
    """
    Get GPU memory usage information
    """
    if cuda.is_available():
        allocated = cuda.memory_allocated() / 1024**3
        reserved = cuda.memory_reserved() / 1024**3
        max_allocated = cuda.max_memory_allocated() / 1024**3
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated
        }
    return None

def checkgpu(gpu_index=0, interval=1):
    """
    实时查看GPU占用率，类似checkcpu的形式
    
    Args:
        gpu_index: GPU设备索引，默认为0
        interval: 采样间隔（秒），用于计算平均利用率，默认为1秒
    
    Example:
        checkgpu()  # 查看GPU 0的状态
        checkgpu(gpu_index=1)  # 查看GPU 1的状态
    """
    if not GPU_MONITORING_AVAILABLE:
        print("GPU监控不可用，请安装: pip install nvidia-ml-py")
        return
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        
        # 获取GPU名称
        gpu_name_bytes = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name_bytes, bytes):
            gpu_name = gpu_name_bytes.decode('utf-8')
        else:
            gpu_name = str(gpu_name_bytes)
        
        # 获取GPU利用率（需要采样间隔来计算平均利用率）
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu
        
        # 获取内存信息
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_gb = memory_info.used / 1024**3
        memory_total_gb = memory_info.total / 1024**3
        memory_available_gb = memory_info.free / 1024**3
        memory_percent = (memory_info.used / memory_info.total) * 100
        
        # 获取温度
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = "N/A"
        
        # 获取功耗
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
        except:
            power = "N/A"
        
        # 打印信息（类似checkcpu的格式）
        print(f"GPU {gpu_index} ({gpu_name}):")
        print(f"  GPU 使用率: {gpu_util}%")
        print(f"  已用内存: {memory_used_gb:.2f} GB")
        print(f"  可用内存: {memory_available_gb:.2f} GB")
        print(f"  总内存: {memory_total_gb:.2f} GB ({memory_percent:.1f}%)")
        print(f"  温度: {temperature}°C" if temperature != "N/A" else f"  温度: {temperature}")
        print(f"  功耗: {power:.1f}W" if power != "N/A" else f"  功耗: {power}")
        
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
