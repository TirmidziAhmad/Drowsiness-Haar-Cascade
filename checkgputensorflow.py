import tensorflow as tf

# List available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if not gpus:
    print("No GPUs found. TensorFlow is using the CPU.")
else:
    print("Available GPUs:")
    for gpu in gpus:
        print(f"- {gpu.name}")

# Check if TensorFlow is running on GPU
if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print("TensorFlow is running on GPU.")
else:
    print("TensorFlow is running on the CPU.")