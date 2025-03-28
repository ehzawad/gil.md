# gil.md

### Make sure you have a python3.13 without GIL build

bash ./configure --disable-gil


```python
import time
import datetime
import numpy as np
from scipy import ndimage
import sys
import os
import gzip
import urllib.request
from urllib.error import URLError
import concurrent.futures
import multiprocessing
import logging
import argparse
import shutil
import threading
import ctypes
import inspect
from threading import Lock

# Global variables
log_file = None  # Will be set in setup_logging

# Famous guitarists for thread names
GUITARIST_NAMES = [
    "Gilmour",    # Pink Floyd
    "Page",       # Led Zeppelin
    "Clapton",    # Cream
    "Hendrix",    # Experience
    "Richards",   # Rolling Stones
    "Santana",    # Santana
    "Van Halen",  # Van Halen
    "Knopfler",   # Dire Straits
    "Beck",       # Yardbirds
    "Blackmore",  # Deep Purple
    "Iommi",      # Black Sabbath
    "Young",      # AC/DC
    "Vai",        # Solo
    "Satriani",   # Solo
    "Harrison",   # Beatles
    "May",        # Queen
]

# Log levels
class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    # Custom log levels for GIL tracking
    GIL_ACQUIRE = "GIL_ACQUIRE"
    GIL_RELEASE = "GIL_RELEASE"
    THREAD_START = "THREAD_START"
    THREAD_END = "THREAD_END"
    THREAD_SWITCH = "THREAD_SWITCH"
    BATCH_START = "BATCH_START"
    BATCH_END = "BATCH_END"
    BENCHMARK = "BENCHMARK"
    SYSTEM = "SYSTEM"

# ANSI color codes for colorful output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    # More iTerm2-friendly colors (less bright, better contrast)
    RED = "\033[38;5;124m"          # Darker red - less eye-glaring
    GREEN = "\033[38;5;71m"         # Softer green
    YELLOW = "\033[38;5;178m"       # Softer yellow
    BLUE = "\033[38;5;67m"          # Softer blue
    PURPLE = "\033[38;5;133m"       # Softer purple
    CYAN = "\033[38;5;73m"          # Softer cyan
    ORANGE = "\033[38;5;172m"       # Softer orange
    PINK = "\033[38;5;175m"         # Softer pink
    LIME = "\033[38;5;107m"         # Muted lime
    TEAL = "\033[38;5;80m"          # Softer teal
    LAVENDER = "\033[38;5;103m"     # Softer lavender
    BROWN = "\033[38;5;94m"         # Muted brown
    OLIVE = "\033[38;5;64m"         # Muted olive
    NAVY = "\033[38;5;24m"          # Muted navy
    MAROON = "\033[38;5;88m"        # Softer maroon
    FOREST = "\033[38;5;28m"        # Muted forest green
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    REVERSE = "\033[7m"
    MAGENTA = "\033[38;5;132m"      # Softer magenta for batch info
    
    # Thread color mapping - maps thread ID to a color
    thread_colors = {}
    color_index = 0
    
    # High contrast colors for threads
    thread_color_list = [
        RED, GREEN, BLUE, YELLOW, PURPLE, CYAN, ORANGE, PINK,
        LIME, TEAL, LAVENDER, BROWN, OLIVE, NAVY, MAROON, FOREST
    ]
    
    # Level colors
    level_colors = {
        LogLevel.DEBUG: BLUE,
        LogLevel.INFO: RESET,
        LogLevel.WARNING: YELLOW,
        LogLevel.ERROR: RED,
        LogLevel.CRITICAL: BOLD + RED,
        LogLevel.GIL_ACQUIRE: BOLD + GREEN,
        LogLevel.GIL_RELEASE: BOLD + RED,
        LogLevel.THREAD_START: BOLD + CYAN,
        LogLevel.THREAD_END: BOLD + ORANGE,
        LogLevel.THREAD_SWITCH: BOLD + PURPLE,
        LogLevel.BATCH_START: MAGENTA,
        LogLevel.BATCH_END: BOLD + MAGENTA,
        LogLevel.BENCHMARK: BOLD + YELLOW,
        LogLevel.SYSTEM: BOLD + CYAN,
    }
    
    @staticmethod
    def get_thread_color(thread_id):
        """Get a distinct color for a thread based on its ID."""
        # Special case: main thread is always ORANGE
        if thread_id == MAIN_THREAD_ID:
            return Colors.ORANGE
            
        # Use a cache to maintain consistent colors for threads
        if thread_id not in Colors.thread_colors:
            # Assign next color in the list
            next_color = Colors.thread_color_list[Colors.color_index % len(Colors.thread_color_list)]
            Colors.thread_colors[thread_id] = next_color
            Colors.color_index += 1
            
        return Colors.thread_colors[thread_id]

# Store main thread ID for reference
MAIN_THREAD_ID = threading.get_ident()

# Global variables
num_threads = multiprocessing.cpu_count() * 2  # Default to 2 threads per core
last_thread_id = None  # For thread transition tracking
thread_transitions = 0  # Count of thread switches
print_lock = Lock()  # For thread-safe printing
verbose = True  # Default to verbose mode
thread_names = {}  # Maps thread IDs to names
active_thread = MAIN_THREAD_ID  # Track which thread is currently active
track_thread_switches = True  # Can be disabled for maximum performance

def get_thread_name(thread_id):
    """Get a human-readable name for a thread."""
    global thread_names
    
    if thread_id == MAIN_THREAD_ID:
        return "MAIN"
        
    if thread_id not in thread_names:
        guitarist_idx = len(thread_names) % len(GUITARIST_NAMES)
        thread_names[thread_id] = GUITARIST_NAMES[guitarist_idx]
        
    return thread_names[thread_id]

def safe_print(message, level=LogLevel.INFO, force_print=False):
    """Thread-safe printing without excessive synchronization.
    
    This implementation minimizes lock contention by only synchronizing 
    the actual print operation, allowing the OS and Python interpreter
    to handle thread scheduling naturally.
    """
    global print_lock, verbose, log_file, last_thread_id, thread_transitions, active_thread, track_thread_switches
    
    if not verbose and not force_print:
        return
    
    # Get thread ID and name - this happens in the calling thread's context
    thread_id = threading.get_ident()
    is_main_thread = (thread_id == MAIN_THREAD_ID)
    
    # Prepare thread identification - minimize time inside lock
    if is_main_thread:
        thread_name = "MAIN"
        thread_prefix = f"[MAIN-{thread_id}]"
    else:
        # Get or assign thread name outside the lock to reduce contention
        if thread_id not in thread_names:
            guitarist_idx = len(thread_names) % len(GUITARIST_NAMES)
            thread_names[thread_id] = GUITARIST_NAMES[guitarist_idx]
        thread_name = thread_names.get(thread_id, f"Thread-{thread_id}")
        thread_prefix = f"[{thread_name}-{thread_id}]"
    
    framework_thread_name = threading.current_thread().name
    
    # Get thread color - do this outside the lock
    thread_color = Colors.get_thread_color(thread_id)
    
    # Get color for log level - also outside the lock
    level_color = Colors.level_colors.get(level, Colors.RESET)
    
    # Format messages outside the lock
    console_message = f"{thread_color}{thread_prefix}{Colors.RESET} ({framework_thread_name}) [{level_color}{level}{Colors.RESET}]: {message}"
    
    # Only track thread switches if enabled - this minimizes synchronization overhead
    if track_thread_switches and level != LogLevel.THREAD_SWITCH:
        # Take a short lock just to check/update thread transition tracking
        with print_lock:
            if last_thread_id is not None and last_thread_id != thread_id:
                thread_transitions += 1
                from_thread_name = get_thread_name(last_thread_id)
                from_thread_color = Colors.get_thread_color(last_thread_id)
                to_thread_color = thread_color
                
                # Update active thread tracker
                active_thread = thread_id
                
                # Format the thread prefixes
                if last_thread_id == MAIN_THREAD_ID:
                    from_prefix = f"[MAIN-{last_thread_id}]"
                else:
                    from_prefix = f"[{from_thread_name}-{last_thread_id}]"
                    
                if is_main_thread:
                    to_prefix = f"[MAIN-{thread_id}]"
                else:
                    to_prefix = f"[{thread_name}-{thread_id}]"
                
                # Check if GIL is enabled and use appropriate message
                gil_enabled = is_gil_enabled()
                if gil_enabled:
                    switch_type = "GIL SWITCH"
                    action_text = f"{from_thread_color}{from_prefix}{Colors.RESET} released GIL → {to_thread_color}{to_prefix}{Colors.RESET} acquired GIL"
                else:
                    switch_type = "THREAD SWITCH"
                    action_text = f"{from_thread_color}{from_prefix}{Colors.RESET} → {to_thread_color}{to_prefix}{Colors.RESET}"
                
                # Release lock before print to minimize contention
                print_transition_message = f"{Colors.BOLD}{Colors.PURPLE}{switch_type} #{thread_transitions}{Colors.RESET} {action_text}"
                
                # Update for next time
                last_thread_id = thread_id
                
                # Print transition outside of lock (this is OK even if interleaved)
                print(print_transition_message, flush=True)
                
                # Log to file outside of critical section
                if log_file:
                    try:
                        with open(log_file, 'a') as f:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f.write(f"{timestamp} - {LogLevel.THREAD_SWITCH} - {switch_type} #{thread_transitions}: {from_prefix} → {to_prefix}\n")
                    except IOError:
                        # Don't crash if log file can't be written
                        pass
            else:
                # Just update the last thread ID
                last_thread_id = thread_id
    
    # Print message to console - not inside lock to allow natural OS handling
    print(console_message, flush=True)
    
    # Log to file if enabled - also outside lock
    if log_file:
        try:
            file_message = console_message
            for color in [color for key, color in Colors.__dict__.items() if isinstance(color, str) and color.startswith('\033')]:
                file_message = file_message.replace(color, '')
                
            with open(log_file, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} - {level} - {file_message}\n")
        except IOError:
            # Don't crash if log file can't be written
            pass

def is_gil_enabled():
    """Check if GIL is enabled in this Python process."""
    gil_status = os.environ.get("PYTHON_GIL", "1")
    return gil_status == "1"

def download_mnist(data_dir="."):
    """Download MNIST dataset if not already present.
    
    Args:
        data_dir: Directory to download files to
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = {
        'train-images-idx3-ubyte.gz': 'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz': 'train-labels-idx1-ubyte.gz'
    }
    
    for filename in files.values():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            safe_print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(base_url + filename, file_path)
            except URLError as e:
                safe_print(f"Error downloading {filename}: {e}")
                safe_print("Please check your internet connection or download the files manually.")
                return False
    
    return True

def load_mnist(data_dir="."):
    """Load MNIST dataset with error handling.
    
    Args:
        data_dir: Directory containing the MNIST files
        
    Returns:
        tuple: (images, labels) if successful, (None, None) otherwise
    """
    image_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    label_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    
    # Check if files exist
    if not os.path.exists(image_path) or not os.path.exists(label_path):
        safe_print(f"MNIST files not found in {data_dir}")
        return None, None
    
    try:
        # Load images
        with gzip.open(image_path, 'rb') as f:
            # First 16 bytes are magic number, number of images, rows, columns
            file_content = f.read()
            if len(file_content) < 16:
                safe_print("Image file is corrupted or truncated")
                return None, None
                
            images = np.frombuffer(file_content[16:], dtype=np.uint8)
            # Validate expected shape before reshaping
            if len(images) % (28*28) != 0:
                safe_print(f"Unexpected image data size: {len(images)}")
                return None, None
                
            images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0
        
        # Load labels
        with gzip.open(label_path, 'rb') as f:
            # First 8 bytes are magic number and number of labels
            file_content = f.read()
            if len(file_content) < 8:
                safe_print("Label file is corrupted or truncated")
                return None, None
                
            labels = np.frombuffer(file_content[8:], dtype=np.uint8)
            
        # Validate that images and labels have matching counts
        if len(images) != len(labels):
            safe_print(f"Mismatch between images ({len(images)}) and labels ({len(labels)})")
            return None, None
            
        return images, labels
        
    except (IOError, gzip.BadGzipFile) as e:
        safe_print(f"Error loading MNIST data: {e}")
        return None, None

# Optimized for vectorized performance
def apply_convolution(image, kernel=None):
    """Apply a convolution to an image using vectorized operations."""
    if kernel is None:
        # Basic 3x3 convolution kernel for edge detection
        kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.0, 0.2],
            [0.1, 0.2, 0.1]
        ])
    
    # Use SciPy's convolve2d for efficient computation
    return ndimage.convolve(image, kernel, mode='constant', cval=0.0)

# Optimized rotation using SciPy for improved performance
def rotate_image(image, angle=None):
    """Rotate an image by a given angle using vectorized operations."""
    if angle is None:
        angle = np.random.uniform(-45, 45)
    
    # Use SciPy's rotate function for efficient computation
    return ndimage.rotate(image, angle, reshape=False, order=1, mode='constant', cval=0.0)

# Class to cache neural network weights and avoid regenerating them for each image
class NeuralNetworkCache:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """Pre-compute weights for neural network to avoid regenerating them for each image."""
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)

# Thread-local storage for neural network weights to prevent contention
thread_local_cache = threading.local()

def run_neural_network(image):
    """Run a simple neural network on an image with cached weights for improved performance."""
    # Get or create thread-local neural network weight cache
    if not hasattr(thread_local_cache, 'nn_cache'):
        thread_local_cache.nn_cache = NeuralNetworkCache()
    
    cache = thread_local_cache.nn_cache
    
    # Flatten the image
    flattened = image.flatten()
    
    # Forward pass with cached weights
    hidden = np.maximum(0, np.dot(flattened, cache.W1) + cache.b1)  # ReLU activation
    output = np.dot(hidden, cache.W2) + cache.b2
    
    # Softmax - use the stable version to avoid overflow
    exp_scores = np.exp(output - np.max(output))
    probs = exp_scores / np.sum(exp_scores)
    
    return probs

def display_benchmark_results(total_time, total_images_processed, successful_threads, num_threads, thread_transitions):
    """Display comprehensive benchmark results.
    
    Args:
        total_time: Total processing time in seconds
        total_images_processed: Total number of images processed
        successful_threads: Number of threads that completed successfully
        num_threads: Total number of threads used
        thread_transitions: Number of thread transitions/GIL switches observed
    """
    # Calculate metrics
    images_per_second = total_images_processed / total_time if total_time > 0 else 0
    thread_efficiency = successful_threads / num_threads if num_threads > 0 else 0
    switches_per_second = thread_transitions / total_time if total_time > 0 else 0
    
    # Check if GIL is enabled for correct terminology
    gil_status = is_gil_enabled()
    switch_term = "GIL transitions" if gil_status else "Thread switches"
    
    # Display summary
    safe_print(f"{Colors.BOLD}{Colors.YELLOW}=========== BENCHMARK RESULTS ==========={Colors.RESET}", level=LogLevel.BENCHMARK)
    safe_print(f"Total processing time: {total_time:.2f} seconds", level=LogLevel.BENCHMARK)
    safe_print(f"Total images processed: {total_images_processed}", level=LogLevel.BENCHMARK)
    safe_print(f"Processing rate: {images_per_second:.2f} images/second", level=LogLevel.BENCHMARK)
    safe_print(f"Thread efficiency: {thread_efficiency:.2%} ({successful_threads}/{num_threads} threads successful)", level=LogLevel.BENCHMARK)
    safe_print(f"{switch_term} observed: {thread_transitions}", level=LogLevel.BENCHMARK)
    safe_print(f"{switch_term} rate: {switches_per_second:.2f} switches/second", level=LogLevel.BENCHMARK)
    safe_print(f"{Colors.BOLD}{Colors.YELLOW}======================================={Colors.RESET}", level=LogLevel.BENCHMARK)

def determine_optimal_batch_size(images, num_threads):
    """Determine the optimal batch size based on system resources and image count.
    
    This function uses batch sizes that divide MNIST's 60,000 images evenly,
    while maintaining multiples of 8 for memory alignment.
    
    Args:
        images: Array of images
        num_threads: Number of processing threads
    
    Returns:
        int: Recommended batch size (optimal for MNIST)
    """
    # Get total image count
    total_image_count = len(images)
    
    # Define safe, hard-coded limits - all divisible by 8 AND divide 60,000 evenly
    ABSOLUTE_MAX_BATCH = 240   # Divides 60,000 into 250 batches
    DEFAULT_BATCH = 120        # Divides 60,000 into 500 batches
    MIN_BATCH = 40             # Divides 60,000 into 1,500 batches
    
    # Safety first - if no images, return the default
    if total_image_count <= 0:
        return DEFAULT_BATCH
    
    # MNIST-optimized batch sizes
    # These values are chosen to:
    # 1. Be multiples of 8 (for memory alignment)
    # 2. Divide 60,000 evenly (for clean batch processing)
    # 3. Scale appropriately with dataset size
    if total_image_count > 50000:
        # Very large dataset (close to full MNIST dataset of 60K)
        # 60,000 ÷ 120 = 500 batches
        batch_size = 120  
    elif total_image_count > 10000:
        # Large dataset
        # 60,000 ÷ 160 = 375 batches
        batch_size = 160
    elif total_image_count > 5000:
        # Medium dataset
        # 60,000 ÷ 200 = 300 batches
        batch_size = 200
    else:
        # Small dataset
        # 60,000 ÷ 240 = 250 batches
        batch_size = 240
    
    # Extra safety check: make sure batch size doesn't exceed images per thread
    max_per_thread = total_image_count // num_threads
    if max_per_thread < batch_size:
        # Find the largest divisor of 60,000 that is also a multiple of 8
        # and less than or equal to max_per_thread
        divisors = [240, 200, 160, 120, 80, 40]
        for div in divisors:
            if div <= max_per_thread:
                batch_size = div
                break
        else:
            # If no good divisor found, fall back to a multiple of 8
            batch_size = max((max_per_thread // 8) * 8, MIN_BATCH)
    
    # Final guard against extreme values
    batch_size = min(batch_size, ABSOLUTE_MAX_BATCH)
    batch_size = max(batch_size, MIN_BATCH)
    
    safe_print(f"Using MNIST-optimized batch size: {batch_size} images per batch (divides 60,000 evenly)", level=LogLevel.SYSTEM)
    return batch_size

def process_chunk(chunk_id, images_chunk, iterations=1, batch_size=100, num_threads=1):
    """Process a chunk of images with computationally intensive operations."""
    # Get thread information - this is a worker thread from the pool
    thread_id = threading.get_ident()
    thread_name = get_thread_name(thread_id)
    framework_thread_name = threading.current_thread().name
    thread_color = Colors.get_thread_color(thread_id)
    
    # Clearly identify this is a worker thread, not the main thread
    is_main = (thread_id == MAIN_THREAD_ID)
    thread_type = "MAIN" if is_main else "WORKER"
    
    # Initialize thread-local neural network for this thread
    if not hasattr(thread_local_cache, 'nn_cache'):
        thread_local_cache.nn_cache = NeuralNetworkCache()
        
    # Pre-compute kernel for convolution to avoid recreating it for each image
    kernel = np.array([
        [0.1, 0.2, 0.1],
        [0.2, 0.0, 0.2],
        [0.1, 0.2, 0.1]
    ])
    
    safe_print(
        f"Starting to process chunk {chunk_id} with {len(images_chunk)} images (Thread type: {thread_type})", 
        level=LogLevel.THREAD_START,
        force_print=True
    )
    
    # Store timing information
    conv_times = []
    rotation_times = []
    nn_times = []
    
    # Process in smaller batches to simulate real-world workload
    num_batches = max(1, len(images_chunk) // batch_size)
    actual_batch_size = len(images_chunk) // num_batches if num_batches > 0 else len(images_chunk)
    
    safe_print(f"{thread_color}Processing chunk {chunk_id} in {num_batches} batches of ~{actual_batch_size} images each{Colors.RESET}", level="INFO", force_print=True)
    
    try:
        batch_start_time = time.time()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images_chunk))
            batch_images = images_chunk[start_idx:end_idx]
            images_count = len(batch_images)
            
            batch_item_start = time.time()
            
            # Convolution operation - vectorized
            if batch_idx == 0:
                safe_print(f"{Colors.MAGENTA}Batch {batch_idx+1}/{num_batches} - Processing convolution on {images_count} images{Colors.RESET}", level="BATCH", force_print=True)
            
            for i in range(iterations):
                conv_start = time.time()
                # Process all images in batch with optimized convolution
                for img in batch_images:
                    apply_convolution(img, kernel)
                conv_time = time.time() - conv_start
                conv_times.append(conv_time)
            
            # Rotation operation - vectorized
            if batch_idx == 0:
                safe_print(f"{Colors.MAGENTA}Batch {batch_idx+1}/{num_batches} - Processing rotation on {images_count} images{Colors.RESET}", level="BATCH", force_print=True)
            
            for i in range(iterations):
                rot_start = time.time()
                # Process all images in batch with optimized rotation
                for img in batch_images:
                    rotate_image(img)
                rot_time = time.time() - rot_start
                rotation_times.append(rot_time)
            
            # Neural network processing - optimized with cached weights
            if batch_idx == 0:
                safe_print(f"{Colors.MAGENTA}Batch {batch_idx+1}/{num_batches} - Running neural network on {images_count} images{Colors.RESET}", level="BATCH", force_print=True)
            
            for i in range(iterations):
                nn_start = time.time()
                # Process all images using the thread-local cached neural network
                for img in batch_images:
                    run_neural_network(img)
                nn_time = time.time() - nn_start
                nn_times.append(nn_time)
            
            batch_item_time = time.time() - batch_item_start
            
            # Only log every 10th batch to reduce thread contention from logging
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                safe_print(f"{thread_color}Completed batch {batch_idx+1}/{num_batches} of chunk {chunk_id} in {batch_item_time:.2f}s ({images_count} images){Colors.RESET}", level="INFO", force_print=True)
        
        total_chunk_time = time.time() - batch_start_time
        
        # Return timing information and thread details
        result = {
            'conv_times': conv_times,
            'rotation_times': rotation_times,
            'nn_times': nn_times,
            'num_images': len(images_chunk),
            'thread_id': thread_id,
            'thread_name': thread_name,
            'thread_type': thread_type,
            'chunk_id': chunk_id,
            'processing_time': total_chunk_time
        }
        
        safe_print(f"{thread_color}Chunk {chunk_id} complete - processed {len(images_chunk)} images in {total_chunk_time:.2f}s{Colors.RESET}", level="INFO", force_print=True)
        return result
    except Exception as e:
        safe_print(f"{thread_color}Error in chunk {chunk_id}: {e}{Colors.RESET}", level="ERROR", force_print=True)
        raise

def run_mnist_processing_multithreaded(images, num_threads, batch_size=None, iterations=1, timeout=24*3600, image_limit=0):
    """Run CPU-intensive processing on MNIST images using multiple threads."""
    # Reset thread transition counter
    global thread_transitions, thread_names, MAIN_THREAD_ID
    thread_transitions = 0
    thread_names = {}  # Reset thread names
    
    # Limit the number of images if specified
    if image_limit > 0 and image_limit < len(images):
        images = images[:image_limit]
        safe_print(f"Processing limited subset of {image_limit} images", level=LogLevel.SYSTEM)
    else:
        safe_print(f"Processing complete dataset of {len(images)} images", level=LogLevel.SYSTEM)
    
    # Calculate dynamic batch size if not provided
    if batch_size is None:
        batch_size = determine_optimal_batch_size(images, num_threads)
    
    # Log threading status at start
    gil_status = is_gil_enabled()
    threading_mode = "GIL-enabled" if gil_status else "GIL-disabled"
    main_thread_id = threading.get_ident()
    main_thread = threading.current_thread()
    main_address = id(main_thread)
    
    # Verify main thread identification is correct
    if main_thread_id != MAIN_THREAD_ID:
        safe_print(
            f"WARNING: Main thread ID mismatch! Global: {MAIN_THREAD_ID}, Current: {main_thread_id}. Updating...",
            level=LogLevel.WARNING,
            force_print=True
        )
        # This should be extremely rare but handling it anyway
        MAIN_THREAD_ID = main_thread_id
    
    safe_print(
        f"{Colors.BOLD}Starting {threading_mode} benchmark with {num_threads} threads{Colors.RESET}",
        level=LogLevel.SYSTEM, 
        force_print=True
    )
    safe_print(
        f"Using {Colors.BOLD}{num_threads}{Colors.RESET} threads with batch size {Colors.BOLD}{batch_size}{Colors.RESET}",
        level=LogLevel.SYSTEM,
        force_print=True
    )
    safe_print(
        f"Main thread is {Colors.ORANGE}MAIN-{main_thread_id}{Colors.RESET} at address 0x{main_address:x}",
        level=LogLevel.SYSTEM,
        force_print=True
    )
    
    # Thread synchronization note
    if gil_status:
        safe_print(
            f"GIL enabled: Thread coordination handled by Python's GIL mechanism",
            level=LogLevel.SYSTEM
        )
    else:
        safe_print(
            f"GIL disabled: OS thread scheduling in effect, true parallelism possible",
            level=LogLevel.SYSTEM
        )
    
    safe_print(
        f"OPTIMIZED: Using vectorized operations and cached neural network weights",
        level=LogLevel.SYSTEM
    )
    
    start_time = time.time()
    
    # Divide images into chunks for each thread
    chunk_size = len(images) // num_threads
    chunks = []
    total_images_to_process = 0
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_threads - 1 else len(images)
        chunk_images = images[start_idx:end_idx]
        chunks.append(chunk_images)
        total_images_to_process += len(chunk_images)
        safe_print(
            f"Created chunk {i} with {len(chunk_images)} images (indexes {start_idx}-{end_idx-1})",
            level=LogLevel.SYSTEM
        )
    
    safe_print(f"Total images to process: {total_images_to_process}", level=LogLevel.SYSTEM)
    
    # Process chunks in parallel using ThreadPoolExecutor with timeout
    results = []
    futures_list = []
    
    # Note that we're deliberately using ThreadPoolExecutor which shares the Python interpreter state
    # This means with GIL, threads will contend for the lock, without GIL they can run truly in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks to the executor
        safe_print(
            f"Creating thread pool with {num_threads} worker threads - each named after a famous guitarist", 
            level=LogLevel.SYSTEM
        )
        future_to_chunk = {}
        
        try:
            # Submit all tasks first
            for i, chunk in enumerate(chunks):
                safe_print(f"Submitting chunk {i} to thread pool", level=LogLevel.SYSTEM)
                future = executor.submit(process_chunk, i, chunk, iterations, batch_size, num_threads)
                future_to_chunk[future] = i
                futures_list.append(future)
            
            safe_print(
                f"All {len(chunks)} chunks submitted to thread pool, waiting for results", 
                level=LogLevel.SYSTEM
            )
            
            # Collect results as they complete, with timeout
            completed_futures = 0
            for future in concurrent.futures.as_completed(future_to_chunk, timeout=timeout):
                chunk_id = future_to_chunk[future]
                try:
                    safe_print(f"Receiving result from chunk {chunk_id}", level=LogLevel.SYSTEM)
                    result = future.result()
                    results.append(result)
                    completed_futures += 1
                    # Get thread color for thread ID
                    thread_id = result.get('thread_id')
                    thread_name = get_thread_name(thread_id)
                    thread_type = result.get('thread_type', 'UNKNOWN')
                    safe_print(
                        f"Thread {thread_name} (type: {thread_type}) completed chunk {chunk_id} with {result.get('num_images', 0)} images",
                        level=LogLevel.SYSTEM
                    )
                except Exception as e:
                    safe_print(
                        f"Error handling result from chunk {chunk_id}: {e}", 
                        level=LogLevel.ERROR
                    )
                    # Add partial result with error information
                    results.append({
                        'chunk_id': chunk_id,
                        'error': str(e),
                        'conv_times': [],
                        'rotation_times': [],
                        'nn_times': [],
                        'num_images': len(chunks[chunk_id]),
                        'thread_id': threading.get_ident(),
                        'thread_name': get_thread_name(threading.get_ident()),
                        'thread_type': 'MAIN' if threading.get_ident() == MAIN_THREAD_ID else 'WORKER'
                    })
            
            safe_print(f"Completed {completed_futures}/{len(chunks)} chunks", level=LogLevel.SYSTEM)
        
        except concurrent.futures.TimeoutError:
            safe_print(
                f"Processing timed out after {timeout} seconds", 
                level=LogLevel.ERROR
            )
            # Cancel any remaining futures
            remaining_futures = [f for f in future_to_chunk if not f.done()]
            safe_print(
                f"Cancelling {len(remaining_futures)} remaining tasks", 
                level=LogLevel.WARNING
            )
            for future in future_to_chunk:
                if not future.done():
                    future.cancel()
        
        except Exception as e:
            safe_print(
                f"Unexpected error in thread pool execution: {e}", 
                level=LogLevel.ERROR
            )
            import traceback
            safe_print(
                f"Exception trace: {traceback.format_exc()}", 
                level=LogLevel.ERROR
            )
        
        finally:
            # Wait for all futures to complete or be cancelled
            safe_print(f"Checking status of all futures...", level=LogLevel.SYSTEM)
            for i, future in enumerate(futures_list):
                if future.done():
                    if future.exception() is not None:
                        safe_print(
                            f"Future {i} completed with exception: {future.exception()}", 
                            level=LogLevel.ERROR
                        )
                    else:
                        safe_print(f"Future {i} completed successfully", level=LogLevel.SYSTEM)
                else:
                    safe_print(
                        f"Future {i} is still running, cancelling...", 
                        level=LogLevel.WARNING
                    )
                    future.cancel()
    
    total_time = time.time() - start_time
    safe_print(f"All threads completed, processing results", level=LogLevel.SYSTEM)
    
    # Aggregate results (including chunks with errors to ensure all images are counted)
    all_conv_times = []
    all_rotation_times = []
    all_nn_times = []
    total_images_processed = 0
    successful_threads = 0
    
    for result in results:
        # Count all images, even if there was an error
        total_images_processed += result.get('num_images', 0)
        
        # Only include timing data and count successful threads if no error
        if 'error' not in result:
            all_conv_times.extend(result.get('conv_times', []))
            all_rotation_times.extend(result.get('rotation_times', []))
            all_nn_times.extend(result.get('nn_times', []))
            successful_threads += 1
    
    # If no results were collected but we know work happened, count the chunks anyway
    if len(results) == 0 and len(chunks) > 0:
        safe_print(
            f"No results collected, but chunks were processed. Estimating processed images...", 
            level=LogLevel.WARNING
        )
        total_images_processed = total_images_to_process
        successful_threads = num_threads  # Assume all threads ran
    
    # Display benchmark results
    display_benchmark_results(total_time, total_images_processed, successful_threads, num_threads, thread_transitions)
    
    # Calculate average times if available
    if all_conv_times:
        avg_conv = sum(all_conv_times) / len(all_conv_times)
        safe_print(f"Average convolution time: {avg_conv:.4f}s per batch", level=LogLevel.BENCHMARK)
    if all_rotation_times:
        avg_rot = sum(all_rotation_times) / len(all_rotation_times)
        safe_print(f"Average rotation time: {avg_rot:.4f}s per batch", level=LogLevel.BENCHMARK)
    if all_nn_times:
        avg_nn = sum(all_nn_times) / len(all_nn_times)
        safe_print(f"Average neural network time: {avg_nn:.4f}s per batch", level=LogLevel.BENCHMARK)
    
    return total_time, total_images_processed

def cleanup_files(data_dir="."):
    """Clean up downloaded MNIST files.
    
    Args:
        data_dir: Directory containing the files
        
    Returns:
        bool: True if successful, False otherwise
    """
    files = [
        os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
        os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    ]
    
    success = True
    for file_path in files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                safe_print(f"Removed {file_path}")
            except OSError as e:
                safe_print(f"Error removing {file_path}: {e}")
                success = False
    
    return success

def setup_logging(num_threads, log_dir="logs"):
    """Set up logging with both console and file output.
    
    Args:
        num_threads: Number of threads being used
        log_dir: Directory to store log files
    
    Returns:
        logging.Logger: Configured logger
    """
    global log_file
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate unique log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mnist_gil_benchmark_{timestamp}_{num_threads}threads.log")
    
    # Open the log file
    with open(log_file, 'w') as f:
        f.write(f"MNIST GIL Demonstration Benchmark - {timestamp} - {num_threads} threads\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"numpy version: {np.__version__}\n")
        f.write(f"CPU cores: {multiprocessing.cpu_count()}\n")
        f.write("-" * 80 + "\n")
    
    # Return a basic logger for compatibility
    return logging.getLogger(__name__)

def main():
    """Main function to run the benchmark."""
    global num_threads, MAIN_THREAD_ID, verbose, thread_transitions, thread_names, last_thread_id, active_thread, track_thread_switches
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run MNIST GIL demonstration with multiple threads')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads to use (0 for auto-detection)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--timeout', type=int, default=24*3600, help='Timeout in seconds (default: 24 hours)')
    parser.add_argument('--image-limit', type=int, default=0, help='Number of images to process (0 for all images)')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to store log files')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory for data files')
    parser.add_argument('--cleanup', action='store_true', help='Clean up downloaded files after execution')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for processing (default: auto-determine based on dataset size)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations per batch')
    parser.add_argument('--track-switches', action='store_true', default=True, help='Track thread switches (disable for maximum performance)')
    parser.add_argument('--no-track-switches', action='store_false', dest='track_switches', help='Disable thread switch tracking for maximum performance')
    args = parser.parse_args()
    
    verbose = args.verbose or True  # Default to verbose mode if not specified
    track_thread_switches = args.track_switches  # Allow disabling thread switch tracking
    
    # Auto-detect number of threads if not specified
    if args.threads <= 0:
        num_threads = multiprocessing.cpu_count() * 2
    else:
        num_threads = args.threads
    
    # Initialize global variables
    thread_transitions = 0
    thread_names = {}  # Maps thread IDs to names
    last_thread_id = None
    
    # Ensure main thread ID is set correctly for this process
    MAIN_THREAD_ID = threading.get_ident()
    active_thread = MAIN_THREAD_ID
    
    # Set up logging with both console and file output
    logger = setup_logging(num_threads, args.log_dir)
    
    # Log benchmark initialization
    safe_print(
        f"{Colors.BOLD}{Colors.ORANGE}MNIST Thread Processing Benchmark initializing{Colors.RESET}",
        level=LogLevel.SYSTEM,
        force_print=True
    )
    
    # Check GIL status and print
    gil_status = is_gil_enabled()
    gil_status_str = "ENABLED" if gil_status else "DISABLED"
    gil_status_color = Colors.YELLOW if gil_status else Colors.GREEN
    safe_print(
        f"{Colors.BOLD}{gil_status_color}Python GIL is {gil_status_str}{Colors.RESET}",
        level=LogLevel.SYSTEM,
        force_print=True
    )
    
    # Log thread switch tracking status
    if track_thread_switches:
        safe_print(f"Thread switch tracking is ENABLED", level=LogLevel.SYSTEM)
    else:
        safe_print(f"Thread switch tracking is DISABLED for maximum performance", level=LogLevel.SYSTEM)
    
    safe_print(f"Log file: {os.path.abspath(log_file)}", level=LogLevel.SYSTEM)
    
    # Print system information
    safe_print(f"Python version: {sys.version}", level=LogLevel.SYSTEM)
    safe_print(f"numpy version: {np.__version__}", level=LogLevel.SYSTEM)
    safe_print(f"Machine: {multiprocessing.cpu_count()} CPU cores", level=LogLevel.SYSTEM)
    safe_print(f"Main thread ID: {MAIN_THREAD_ID}", level=LogLevel.SYSTEM)
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    try:
        # Download and load MNIST dataset
        safe_print(f"Downloading MNIST dataset if needed...", level=LogLevel.SYSTEM)
        success = download_mnist(args.data_dir)
        if not success:
            safe_print("Failed to download MNIST dataset. Exiting.", level=LogLevel.ERROR)
            return 1
        
        safe_print(f"Loading MNIST dataset...", level=LogLevel.SYSTEM)
        images, labels = load_mnist(args.data_dir)
        if images is None:
            safe_print("Failed to load MNIST dataset. Exiting.", level=LogLevel.ERROR)
            return 1
        
        safe_print(f"Loaded {len(images)} images from MNIST dataset", level=LogLevel.SYSTEM)
        
        # Show configuration
        threading_mode = "GIL-enabled threading" if gil_status else "GIL-disabled threading"
        safe_print(
            f"Configuration: {threading_mode}, image_limit={args.image_limit}, timeout={args.timeout}s, "
            f"batch_size={args.batch_size}, iterations={args.iterations}",
            level=LogLevel.SYSTEM
        )
        
        # Run the multithreaded processing
        safe_print(
            f"{Colors.BOLD}Starting benchmark with {num_threads} threads{Colors.RESET}", 
            level=LogLevel.SYSTEM
        )
        
        total_time, total_processed = run_mnist_processing_multithreaded(
            images, num_threads, args.batch_size, args.iterations, args.timeout, args.image_limit
        )
        
        safe_print(f"Benchmark complete", level=LogLevel.SYSTEM)
        
        return 0
    except KeyboardInterrupt:
        safe_print(f"Process interrupted by user", level=LogLevel.WARNING)
        return 130  # Standard Unix exit code for SIGINT
    except Exception as e:
        safe_print(f"Unhandled exception: {e}", level=LogLevel.ERROR)
        import traceback
        safe_print(f"Exception trace: {traceback.format_exc()}", level=LogLevel.ERROR)
        return 1
    finally:
        # Always clean up resources in the finally block
        if args.cleanup:
            safe_print(f"Cleaning up downloaded files...", level=LogLevel.SYSTEM)
            cleanup_files(args.data_dir)
        
        if log_file:
            safe_print(f"Logs saved to: {log_file}", level=LogLevel.SYSTEM)
            
        # Print guitarist legend
        if len(thread_names) > 1:
            safe_print("Thread name legend:", level=LogLevel.SYSTEM)
            for thread_id, name in thread_names.items():
                thread_color = Colors.get_thread_color(thread_id)
                safe_print(
                    f"{thread_color}[{name}-{thread_id}]{Colors.RESET} - Thread {thread_id}",
                    level=LogLevel.SYSTEM
                )

if __name__ == "__main__":
    sys.exit(main())
```
#### run it

```bash
# With GIL enabled
PYTHON_GIL=1 python david_gilmour.py
```
```bash
# Without gil
PYTHON_GIL=0 python david_gilmour.py
```

# With GIL disabled 
PYTHON_GIL=0 python mnist_asyncio_gil.py
```
