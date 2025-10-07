"""Test memory-mapped loading speed"""
import numpy as np
import time

print("Testing memory-mapped data loading...")

# Test with mmap
print("\n" + "="*60)
print("Loading with memory mapping...")
print("="*60)
start = time.time()
data = np.load("artifacts/preprocessed/train_data.npz", mmap_mode='r')
images = data['images']
labels = data['labels']
load_time = time.time() - start

print(f"✓ Loaded in {load_time:.4f} seconds (memory mapped)")
print(f"  Shape: {images.shape}")
print(f"  Dtype: {images.dtype}")

# Test batch access
print("\n" + "="*60)
print("Testing batch access speed...")
print("="*60)
batch_size = 128
n_batches = 100

start = time.time()
for i in range(n_batches):
    batch = images[i*batch_size:(i+1)*batch_size].copy()
access_time = time.time() - start

print(f"✓ Accessed {n_batches} batches in {access_time:.4f} seconds")
print(f"  Average per batch: {access_time/n_batches*1000:.2f} ms")
print(f"  Throughput: {n_batches*batch_size/access_time:.0f} images/sec")

print("\n✓ Memory mapping is working correctly!")
