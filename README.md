# gil.md

### Make sure you have a python3.13 without GIL build
```bash ./configure --disable-gil
```

```python
import time
import numpy as np
import sys
import os
import gzip
import urllib.request
from urllib.error import URLError

def is_gil_enabled():
    """Check if GIL is enabled in this Python process."""
    gil_status = os.environ.get("PYTHON_GIL", "1")
    return gil_status == "1"

def download_mnist():
    """Download MNIST dataset if not already present."""
    base_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = {
        'train-images-idx3-ubyte.gz': 'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz': 'train-labels-idx1-ubyte.gz'
    }
    
    for filename in files.values():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(base_url + filename, filename)
            except URLError as e:
                print(f"Error downloading {filename}: {e}")
                print("Please check your internet connection or download the files manually.")
                sys.exit(1)

def load_mnist():
    """Load MNIST dataset."""
    with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
        # First 16 bytes are magic number, number of images, rows, columns
        images = np.frombuffer(f.read()[16:], dtype=np.uint8)
        images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0
    
    with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
        # First 8 bytes are magic number and number of labels
        labels = np.frombuffer(f.read()[8:], dtype=np.uint8)
    
    return images, labels

def convolve2d(image, kernel):
    """Manual 2D convolution implementation (CPU-intensive)."""
    # Get dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Output dimensions
    o_height = i_height - k_height + 1
    o_width = i_width - k_width + 1
    
    # Initialize output
    output = np.zeros((o_height, o_width))
    
    # Perform convolution with more operations
    for y in range(o_height):
        for x in range(o_width):
            # Extract the current window
            window = image[y:y+k_height, x:x+k_width]
            # Perform element-wise multiplication and sum
            # Add more CPU-intensive operations
            result = np.sum(window * kernel)
            result = np.sin(result) + np.cos(result)
            result = np.power(result, 0.99)
            output[y, x] = result
    
    return output

def apply_multiple_filters(image, num_filters=20):
    """Apply multiple random filters to an image (CPU-intensive)."""
    results = []
    
    for _ in range(num_filters):
        # Create a random 7x7 filter (increased from 5x5)
        kernel = np.random.randn(7, 7)
        # Normalize the kernel
        kernel = kernel / np.sum(np.abs(kernel))
        # Apply convolution
        filtered = convolve2d(image, kernel)
        # Add more CPU-intensive operations
        filtered = np.power(filtered, 0.99)
        filtered = np.sin(filtered) + np.cos(filtered)
        results.append(filtered)
    
    return results

def rotate_image(image, angle):
    """Rotate image by angle degrees (CPU-intensive)."""
    # Get image dimensions
    height, width = image.shape
    
    # Create rotation matrix
    angle_rad = np.radians(angle)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    
    # Calculate new dimensions
    new_height = int(abs(height * cos_val) + abs(width * sin_val)) + 1
    new_width = int(abs(width * cos_val) + abs(height * sin_val)) + 1
    
    # Create output image
    output = np.zeros((new_height, new_width))
    
    # Compute center of rotation
    center_y, center_x = height // 2, width // 2
    new_center_y, new_center_x = new_height // 2, new_width // 2
    
    # Perform rotation manually
    for y in range(new_height):
        for x in range(new_width):
            # Move to origin
            y_centered = y - new_center_y
            x_centered = x - new_center_x
            
            # Rotate
            original_y = int(y_centered * cos_val - x_centered * sin_val + center_y)
            original_x = int(y_centered * sin_val + x_centered * cos_val + center_x)
            
            # Check if the point is valid
            if 0 <= original_y < height and 0 <= original_x < width:
                output[y, x] = image[original_y, original_x]
    
    return output

def custom_neural_network(images, num_images=1000, hidden_size=256):  # Increased hidden size
    """Simulate a neural network forward pass (CPU-intensive)."""
    # Create random weights with larger matrices
    W1 = np.random.randn(28*28, hidden_size) * 0.01
    W2 = np.random.randn(hidden_size, hidden_size) * 0.01  # Added another hidden layer
    W3 = np.random.randn(hidden_size, 10) * 0.01
    
    predictions = []
    
    for i in range(min(num_images, len(images))):
        # Flatten image
        x = images[i].reshape(28*28)
        
        # Forward pass with more layers and operations
        # First layer with ReLU activation
        z1 = np.dot(x, W1)
        a1 = np.maximum(0, z1)  # ReLU
        a1 = np.sin(a1) + np.cos(a1)  # Additional operations
        
        # Second layer
        z2 = np.dot(a1, W2)
        a2 = np.maximum(0, z2)  # ReLU
        a2 = np.power(a2, 0.99)  # Additional operations
        
        # Third layer
        z3 = np.dot(a2, W3)
        
        # Softmax with more operations
        exp_scores = np.exp(z3 - np.max(z3))
        probs = exp_scores / np.sum(exp_scores)
        probs = np.power(probs, 0.99)  # Additional operations
        
        predictions.append(probs)
    
    return np.array(predictions)

def process_batch(images, batch_idx, batch_size, iterations):
    """Process a batch of images with CPU-intensive operations."""
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(images))
    batch_images = images[start_idx:end_idx]
    
    total_time = 0
    
    # Repeat operations to make it take longer
    for iteration in range(iterations):
        # 1. Apply convolutions
        t0 = time.time()
        
        for i in range(min(20, len(batch_images))):  # Increased from 5 to 20
            filtered_results = apply_multiple_filters(batch_images[i], num_filters=20)  # Increased filters
        
        t1 = time.time()
        conv_time = t1 - t0
        
        # 2. Apply rotations
        t0 = time.time()
        
        rotated_images = []
        for i in range(min(50, len(batch_images))):  # Increased from 20 to 50
            angle = np.random.uniform(-45, 45)
            rotated = rotate_image(batch_images[i], angle)
            # Add more CPU-intensive operations
            rotated = np.power(rotated, 0.99)
            rotated = np.sin(rotated) + np.cos(rotated)
            rotated_images.append(rotated)
        
        t1 = time.time()
        rotation_time = t1 - t0
        
        # 3. Run neural network forward pass
        t0 = time.time()
        
        nn_predictions = custom_neural_network(batch_images, num_images=min(200, len(batch_images)))  # Increased from 100 to 200
        
        t1 = time.time()
        nn_time = t1 - t0
        
        # Calculate batch time
        batch_time = conv_time + rotation_time + nn_time
        total_time += batch_time
        
        print(f"  Batch {batch_idx+1}, Iter {iteration+1}: Conv: {conv_time:.2f}s, Rotation: {rotation_time:.2f}s, NN: {nn_time:.2f}s")
    
    return total_time, conv_time, rotation_time, nn_time

def run_mnist_processing(images, num_batches, batch_size, iterations):
    """Run CPU-intensive processing on MNIST images."""
    gil_status = "enabled" if is_gil_enabled() else "disabled"
    print(f"\n=== Running MNIST processing with GIL {gil_status} ===")
    
    start_time = time.time()
    
    batch_times = []
    conv_times = []
    rotation_times = []
    nn_times = []
    
    for batch_idx in range(num_batches):
        print(f"\nProcessing batch {batch_idx+1}/{num_batches}...")
        
        batch_time, conv_time, rotation_time, nn_time = process_batch(
            images, batch_idx, batch_size, iterations
        )
        
        batch_times.append(batch_time)
        conv_times.append(conv_time)
        rotation_times.append(rotation_time)
        nn_times.append(nn_time)
    
    total_time = time.time() - start_time
    
    # Print performance summary
    print("\n=== Performance Summary ===")
    print(f"GIL: {'Enabled' if is_gil_enabled() else 'Disabled'}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average batch time: {np.mean(batch_times):.2f} seconds")
    print(f"Average convolution time: {np.mean(conv_times):.2f} seconds")
    print(f"Average rotation time: {np.mean(rotation_times):.2f} seconds")
    print(f"Average neural network time: {np.mean(nn_times):.2f} seconds")
    
    return total_time

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"GIL status: {'Enabled' if is_gil_enabled() else 'Disabled'}")
    
    # Download and load MNIST dataset
    print("Downloading MNIST dataset if needed...")
    download_mnist()
    print("Loading MNIST dataset...")
    images, labels = load_mnist()
    print(f"Loaded {len(images)} images")
    
    # Configuration (adjusted for more intensive processing)
    NUM_BATCHES = 3        # Reduced from 5 to 3
    BATCH_SIZE = 200       # Increased from 100 to 200
    ITERATIONS = 5         # Increased from 3 to 5
    
    # Run the processing
    total_time = run_mnist_processing(images, NUM_BATCHES, BATCH_SIZE, ITERATIONS)
    
    # Instructions for comparing GIL vs no-GIL
    print("\nTo compare GIL vs no-GIL performance:")
    print("1. With GIL (default Python): python mnist_gil_test.py")
    print("2. Without GIL (Python 3.12+): PYTHON_GIL=0 python mnist_gil_test.py")
    
    # If this run was too fast or too slow, suggest adjustments
    if total_time < 120:  # Less than 2 minutes
        print("\nThis run completed quickly. For a more noticeable difference:")
        print("  - Increase ITERATIONS (currently {ITERATIONS})")
        print("  - Increase NUM_BATCHES (currently {NUM_BATCHES})")
    elif total_time > 600:  # More than 10 minutes
        print("\nThis run took longer than expected. For a shorter runtime:")
        print("  - Decrease ITERATIONS (currently {ITERATIONS})")
        print("  - Decrease NUM_BATCHES (currently {NUM_BATCHES})") 

```
#### run it
```bash
# With GIL enabled
python mnist_asyncio_gil.py

# With GIL disabled 
PYTHON_GIL=0 python mnist_asyncio_gil.py
```
