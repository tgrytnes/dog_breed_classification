"""Quick test of preprocessed data loading speed"""
import numpy as np
import time
from pathlib import Path

print("Testing preprocessed data loading speed...")

# Test loading train data
print("\n" + "="*60)
print("Loading training data...")
print("="*60)
start = time.time()
train_data = np.load("artifacts/preprocessed/train_data.npz")
train_images = train_data['images']
train_labels = train_data['labels']
load_time = time.time() - start

print(f"✓ Loaded in {load_time:.2f} seconds")
print(f"  Shape: {train_images.shape}")
print(f"  Memory: {train_images.nbytes / (1024**3):.2f} GB")

# Test loading val data
print("\n" + "="*60)
print("Loading validation data...")
print("="*60)
start = time.time()
val_data = np.load("artifacts/preprocessed/val_data.npz")
val_images = val_data['images']
val_labels = val_data['labels']
load_time = time.time() - start

print(f"✓ Loaded in {load_time:.2f} seconds")
print(f"  Shape: {val_images.shape}")
print(f"  Memory: {val_images.nbytes / (1024**3):.2f} GB")

print("\n" + "="*60)
print("Testing batch access speed...")
print("="*60)

# Test batch access speed
batch_size = 128
n_batches = 10
start = time.time()
for i in range(n_batches):
    batch_images = train_images[i*batch_size:(i+1)*batch_size]
    batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
access_time = time.time() - start

print(f"✓ Accessed {n_batches} batches in {access_time:.4f} seconds")
print(f"  Average per batch: {access_time/n_batches*1000:.2f} ms")

print("\n" + "="*60)
print("SUCCESS! Preprocessed data is working correctly.")
print("="*60)
