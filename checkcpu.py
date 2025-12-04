import psutil
print("CPU 使用率:", psutil.cpu_percent(interval=1))
print("可用内存:", psutil.virtual_memory().available / 1024**3, "GB")
