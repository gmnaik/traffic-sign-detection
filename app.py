import torch
import time

cpu = torch.device("cpu")

# Create large tensors
x_cpu = torch.randn(10000, 10000).to(cpu)

# Perform matrix multiplication on CPU
start_time = time.time()
y_cpu = x_cpu @ x_cpu
end_time = time.time()

print(f"Matrix multiplication on CPU took {end_time - start_time:.4f} seconds")
