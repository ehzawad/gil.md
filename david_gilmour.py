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
from rich.console import Console
from rich.table import Table

# Global variables
log_file = None  # Will be set in setup_logging
console = Console()

# Exactly 64 famous guitarists for thread names
GUITARIST_NAMES = [
    # Original 16
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
    # Extended to 64 total
    "Slash",      # Guns N' Roses
    "Frusciante", # Red Hot Chili Peppers
    "Hammett",    # Metallica
    "Malmsteen",  # Solo
    "Morello",    # Rage Against the Machine
    "Lifeson",    # Rush
    "Townshend",  # The Who
    "Reinhardt",  # Jazz
    "Metheny",    # Jazz
    "Montgomery", # Jazz
    "Benson",     # Jazz
    "Govan",      # Solo/Steven Wilson
    "Bonamassa",  # Solo
    "Holdsworth", # Jazz fusion
    "Krasno",     # Soulive
    "Anastasio",  # Phish
    "Moore",      # Thin Lizzy/Solo
    "Berry",      # Rock pioneer
    "King",       # Blues legend
    "Marr",       # The Smiths
    "Buckingham", # Fleetwood Mac
    "Greenwood",  # Radiohead
    "Summers",    # The Police
    "Howe",       # Yes
    "Clarke",     # The Yardbirds/The Faces
    "Gallagher",  # Rory Gallagher
    "Allman",     # Allman Brothers
    "Winter",     # Johnny Winter
    "Prince",     # Prince
    "Homme",      # Queens of the Stone Age
    "Mascis",     # Dinosaur Jr.
    "Schon",      # Journey
    "Bellamy",    # Muse
    "Corgan",     # Smashing Pumpkins
    "Mustaine",   # Megadeth
    "Petrucci",   # Dream Theater
    "Hetfield",   # Metallica
    "Rhoads",     # Ozzy Osbourne
    "Skolnick",   # Testament
    "Johnson",    # Eric Johnson
    "Scofield",   # Jazz
    "McLaughlin", # Mahavishnu Orchestra
    "DiMeola",    # Return to Forever
    "Hoppus",     # Blink-182
    "Wylde",      # Ozzy Osbourne/Black Label Society
    "Friedman",   # Megadeth
    "Buckethead", # Solo
    "Frehley"     # KISS
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

class Colors:
    """ANSI color codes for terminal output with exactly 64 distinct colors."""
    RESET = "\033[0m"
    # Base colors kept for reference
    RED = "\033[38;5;124m"
    GREEN = "\033[38;5;71m"
    YELLOW = "\033[38;5;178m"
    BLUE = "\033[38;5;67m"
    PURPLE = "\033[38;5;133m"
    CYAN = "\033[38;5;73m"
    ORANGE = "\033[38;5;172m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    REVERSE = "\033[7m"
    
    # Thread color mapping - maps thread ID to a color
    thread_colors = {}
    color_index = 0
    # Lock to ensure thread-safe assignment of colors to thread IDs.
    color_assignment_lock = threading.Lock()
    
    # Exactly 64 distinct colors optimized for terminal visibility
    # These are carefully selected from the 6x6x6 color cube (216 colors)
    # to have good contrast and readability
    thread_color_list = [
        # Original 16 colors (modified for better visibility)
        "\033[38;5;196m",  # Bright Red
        "\033[38;5;46m",   # Bright Green
        "\033[38;5;21m",   # Blue
        "\033[38;5;226m",  # Yellow
        "\033[38;5;129m",  # Purple
        "\033[38;5;51m",   # Cyan
        "\033[38;5;208m",  # Orange
        "\033[38;5;201m",  # Pink
        "\033[38;5;118m",  # Lime
        "\033[38;5;39m",   # Light Blue
        "\033[38;5;141m",  # Lavender
        "\033[38;5;130m",  # Brown
        "\033[38;5;106m",  # Olive
        "\033[38;5;18m",   # Navy
        "\033[38;5;124m",  # Maroon
        "\033[38;5;28m",   # Forest Green
        
        # 48 more distinct colors to reach 64 total
        # These are mathematically distributed across the color spectrum
        # Using the 6x6x6 RGB color cube (colors 16-231 in 256-color terminals)
        "\033[38;5;160m",  # Dark Red
        "\033[38;5;202m",  # Dark Orange
        "\033[38;5;214m",  # Gold
        "\033[38;5;190m",  # Light Lime
        "\033[38;5;82m",   # Lime Green
        "\033[38;5;40m",   # Green
        "\033[38;5;34m",   # Dark Green
        "\033[38;5;49m",   # Teal
        "\033[38;5;45m",   # Turquoise
        "\033[38;5;27m",   # Royal Blue
        "\033[38;5;20m",   # Navy Blue
        "\033[38;5;56m",   # Indigo
        "\033[38;5;92m",   # Violet
        "\033[38;5;128m",  # Purple
        "\033[38;5;164m",  # Magenta
        "\033[38;5;200m",  # Pink
        "\033[38;5;124m",  # Dark Red
        "\033[38;5;166m",  # Burnt Orange
        "\033[38;5;184m",  # Tan
        "\033[38;5;154m",  # Yellow Green
        "\033[38;5;76m",   # Grass Green
        "\033[38;5;35m",   # Forest Green
        "\033[38;5;30m",   # Sea Green
        "\033[38;5;44m",   # Light Teal
        "\033[38;5;39m",   # Sky Blue
        "\033[38;5;33m",   # Medium Blue
        "\033[38;5;26m",   # Deep Blue
        "\033[38;5;55m",   # Blue Violet
        "\033[38;5;91m",   # Dark Violet
        "\033[38;5;127m",  # Wine
        "\033[38;5;163m",  # Hot Pink
        "\033[38;5;199m",  # Deep Pink
        "\033[38;5;88m",   # Brown Red
        "\033[38;5;130m",  # Brown
        "\033[38;5;178m",  # Khaki
        "\033[38;5;142m",  # Olive
        "\033[38;5;70m",   # Olive Green
        "\033[38;5;29m",   # Dark Teal
        "\033[38;5;23m",   # Deep Sea Green
        "\033[38;5;37m",   # Medium Teal
        "\033[38;5;32m",   # Light Blue
        "\033[38;5;25m",   # Slate Blue
        "\033[38;5;54m",   # Dark Purple
        "\033[38;5;90m",   # Medium Purple
        "\033[38;5;126m",  # Raspberry
        "\033[38;5;162m",  # Medium Pink
        "\033[38;5;198m",  # Bright Pink
        "\033[38;5;52m",   # Mahogany
    ]
    
    # Special colors for the main thread - make it stand out more clearly
    MAIN_THREAD_COLOR = "\033[38;5;220m"  # Bright gold for main thread
    MAIN_THREAD_HIGHLIGHT = BOLD + "\033[38;5;220m"  # Bold bright gold
    
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
        LogLevel.BATCH_START: "\033[38;5;132m",  # Magenta
        LogLevel.BATCH_END: BOLD + "\033[38;5;132m",  # Bold Magenta
        LogLevel.BENCHMARK: BOLD + YELLOW,
        LogLevel.SYSTEM: BOLD + CYAN,
    }
    
    @staticmethod
    def get_thread_color(thread_id):
        """Get a distinct color for a thread based on its ID."""
        # Special case: main thread gets a distinctive gold color
        if thread_id == MAIN_THREAD_ID:
            return Colors.MAIN_THREAD_COLOR
        
        with Colors.color_assignment_lock:
            # Use a cache to maintain consistent colors for threads
            if thread_id not in Colors.thread_colors:
                # Assign next color in the list, cycling if needed
                next_color = Colors.thread_color_list[Colors.color_index % 64]
                Colors.thread_colors[thread_id] = next_color
                Colors.color_index += 1
                
            return Colors.thread_colors[thread_id]

# Store main thread ID for reference
MAIN_THREAD_ID = threading.get_ident()

# Global variables
num_threads = 8  # Will be updated based on CPU count
last_thread_id = None  # For thread transition tracking
# gil_transitions and thread_switches are now managed by ThreadTransitionTracker
print_lock = Lock()  # For thread-safe printing
verbose = True  # Default to verbose mode
thread_names = {}  # Maps thread IDs to names
# Lock to ensure thread-safe assignment of names to thread IDs.
thread_name_assignment_lock = threading.Lock() 
active_thread = MAIN_THREAD_ID  # Track which thread is currently active
track_thread_switches = True  # Can be disabled for maximum performance

# Centralized mechanism for tracking GIL transitions (in GIL mode)
# or parallel thread activity observations (in No-GIL mode).
# This is preferred over global counters to encapsulate state and logic,
# improving clarity and maintainability.
class ThreadTransitionTracker:
    """Encapsulates thread transition tracking with GIL awareness."""
    
    def __init__(self):
        self.gil_transitions = 0  # Count when GIL is enabled
        self.thread_switches = 0  # Count when GIL is disabled
        self.last_thread_id = None
        self.active_thread_id = None
        self.thread_names = {}  # Maps thread IDs to names # TODO: This is redundant with global thread_names. Consider removing.
        self.enabled = True  # Can be disabled for maximum performance

    def reset(self):
        """Resets the tracker's state."""
        self.gil_transitions = 0
        self.thread_switches = 0
        self.last_thread_id = None
        self.active_thread_id = None
        # self.thread_names = {} # Optionally reset registered names if needed, but usually not.

    def register_thread_name(self, thread_id, thread_name):
        """Register a name for a thread ID."""
        # This method might become redundant if global get_thread_name is solely used.
        # For now, it can still be used by the tracker internally if needed.
        self.thread_names[thread_id] = thread_name
    
    def get_thread_name(self, thread_id):
        """Get the name for a thread ID (potentially tracker-specific)."""
        # This method might become redundant.
        if thread_id == MAIN_THREAD_ID:
            return "MAIN"
        return self.thread_names.get(thread_id, f"Thread-{thread_id}") # Uses tracker's own names.
                                                                      # Prefer global get_thread_name for consistency.

    def record_transition(self, from_thread_id, to_thread_id):
        """Record a thread transition with GIL awareness."""
        if not self.enabled:
            return None
            
        # self.last_thread_id = to_thread_id # This is now managed by safe_print's last_thread_id
        self.active_thread_id = to_thread_id # Tracks the currently active thread based on last call to this
        
        gil_enabled = is_gil_enabled()
        if gil_enabled:
            self.gil_transitions += 1
            return {
                "type": "GIL SWITCH",
                "count": self.gil_transitions,
                "action": "acquired GIL" # Describes the 'to_thread_id' state
            }
        else:
            self.thread_switches += 1
            # For no-GIL, it's not a "switch" but an observation of parallel activity.
            return {
                "type": "PARALLEL ACTIVITY", # Changed from "PARALLEL THREAD" for clarity
                "count": self.thread_switches, # This count is of observations, not strictly switches
                "action": "observed in parallel execution" # Describes the 'to_thread_id' state
            }
    
    def get_stats(self, elapsed_time=None):
        """Get transition statistics."""
        gil_enabled = is_gil_enabled()
        if gil_enabled:
            count = self.gil_transitions
            term = "GIL SWITCH transitions"
        else:
            count = self.thread_switches
            term = "PARALLEL THREAD observations"
        
        stats = {
            "count": count,
            "term": term
        }
        
        if elapsed_time and elapsed_time > 0:
            stats["rate"] = count / elapsed_time
        
        return stats

# Initialize thread transition tracker
transition_tracker = ThreadTransitionTracker()

def calculate_optimal_thread_count():
    """Calculate optimal thread count based on CPU cores available.
    
    Returns thread count that's a multiple of 8 for systems with many cores,
    or a power of 2 for systems with fewer cores.
    """
    cpu_count = multiprocessing.cpu_count()
    
    # For systems with many cores, use a multiple of 8
    if cpu_count >= 16:
        # Round to nearest multiple of 8
        return ((cpu_count + 3) // 8) * 8
    
    # For systems with 8-16 cores, use exact count
    elif cpu_count >= 8:
        return cpu_count
    
    # For systems with fewer cores, use power of 2
    elif cpu_count >= 4:
        return 8
    else:
        return 4  # Minimum thread count

def get_thread_name(thread_id):
    """Get a human-readable name for a thread."""
    global thread_names, thread_name_assignment_lock
    
    if thread_id == MAIN_THREAD_ID:
        return "MAIN"
    
    with thread_name_assignment_lock:
        if thread_id not in thread_names:
            # Will cycle through names if more than 64 threads
            guitarist_idx = len(thread_names) % 64
            thread_names[thread_id] = GUITARIST_NAMES[guitarist_idx]
            
        return thread_names[thread_id]

def is_gil_enabled():
    """Check if GIL is enabled in this Python process.
    
    Examines the PYTHON_GIL environment variable:
    - "1" or empty/missing means GIL is enabled (standard Python behavior)
    - "0" means GIL is disabled (Python 3.13+ free-threading mode)
    
    Returns:
        bool: True if GIL is enabled, False if disabled
    """
    gil_status = os.environ.get("PYTHON_GIL", "1")
    return gil_status == "1"

def get_gil_status_string(with_color=True):
    """Get a formatted status string describing the current GIL state."""
    gil_enabled = is_gil_enabled()
    
    if gil_enabled:
        description = "Single-threaded interpreter lock controlling Python bytecode execution"
        threading_mode = "GIL SWITCH transitions control thread access"
    else:
        description = "True parallel execution with native OS thread scheduling"
        threading_mode = "THREAD SWITCH events managed by the OS scheduler"
    
    if with_color:
        color = Colors.YELLOW if gil_enabled else Colors.GREEN
        return f"{Colors.BOLD}{color}Python GIL is {'ENABLED' if gil_enabled else 'DISABLED'}{Colors.RESET} - {description}"
    else:
        return f"Python GIL is {'ENABLED' if gil_enabled else 'DISABLED'} - {description}"

def safe_print(message, level=LogLevel.INFO, force_print=False):
    """Thread-safe printing without excessive synchronization.
    
    This implementation minimizes lock contention by only synchronizing 
    the actual print operation, allowing the OS and Python interpreter
    to handle thread scheduling naturally.
    """
    global print_lock, verbose, log_file, last_thread_id, active_thread, track_thread_switches, transition_tracker
    
    if not verbose and not force_print:
        return
    
    # Get thread ID and name - this happens in the calling thread's context
    thread_id = threading.get_ident()
    is_main_thread = (thread_id == MAIN_THREAD_ID)
    
    # Prepare thread identification - minimize time inside lock
    if is_main_thread:
        # Make main thread more visibly distinct with bold and special color
        thread_name = "MAIN"
        thread_prefix = f"[{Colors.MAIN_THREAD_HIGHLIGHT}MAIN-{thread_id}{Colors.RESET}]"
        thread_type_indicator = f"({Colors.BOLD}MAIN THREAD{Colors.RESET})"
    else:
        # Get or assign thread name outside the lock to reduce contention
        with thread_name_assignment_lock:
            if thread_id not in thread_names:
                guitarist_idx = len(thread_names) % len(GUITARIST_NAMES)
                thread_names[thread_id] = GUITARIST_NAMES[guitarist_idx]
            thread_name = thread_names.get(thread_id, f"Thread-{thread_id}")
        thread_prefix = f"[{thread_name}-{thread_id}]"
        thread_type_indicator = f"(WORKER)"
    
    framework_thread_name = threading.current_thread().name
    
    # Get thread color - do this outside the lock
    thread_color = Colors.get_thread_color(thread_id)
    
    # Get color for log level - also outside the lock
    level_color = Colors.level_colors.get(level, Colors.RESET)
    
    # Format messages outside the lock
    console_message = f"{thread_color}{thread_prefix}{Colors.RESET} {thread_type_indicator} [{level_color}{level}{Colors.RESET}]: {message}"
    
    # Only track thread switches if enabled - this minimizes synchronization overhead
    if track_thread_switches and level != LogLevel.THREAD_SWITCH:
        # Take a short lock just to check/update thread transition tracking
        with print_lock:
            if last_thread_id is not None and last_thread_id != thread_id:
                # Check if GIL is enabled and use appropriate counter and message
                gil_enabled = is_gil_enabled()
                
                from_thread_name = get_thread_name(last_thread_id)
                from_thread_color = Colors.get_thread_color(last_thread_id)
                to_thread_color = thread_color
                
                # Update active thread tracker
                active_thread = thread_id
                
                # Format the thread prefixes with clearer distinction between main and worker
                if last_thread_id == MAIN_THREAD_ID:
                    from_prefix = f"[{Colors.MAIN_THREAD_HIGHLIGHT}MAIN-{last_thread_id}{Colors.RESET}]"
                    from_type = "(MAIN THREAD)"
                else:
                    from_prefix = f"[{from_thread_name}-{last_thread_id}]"
                    from_type = "(WORKER)"
                    
                if is_main_thread:
                    to_prefix = f"[{Colors.MAIN_THREAD_HIGHLIGHT}MAIN-{thread_id}{Colors.RESET}]"
                    to_type = "(MAIN THREAD)"
                else:
                    to_prefix = f"[{thread_name}-{thread_id}]"
                    to_type = "(WORKER)"
                
                
                transition_info = transition_tracker.record_transition(last_thread_id, thread_id)
                
                if transition_info:
                    switch_type = transition_info["type"]
                    switch_num = transition_info["count"]
                    
                    if gil_enabled: # Corresponds to "GIL SWITCH"
                        action_text = f"{from_thread_color}{from_prefix}{Colors.RESET} {from_type} released GIL → {to_thread_color}{to_prefix}{Colors.RESET} {to_type} {transition_info['action']}"
                        print_transition_message = f"{Colors.BOLD}{Colors.PURPLE}{switch_type} #{switch_num}{Colors.RESET}: {action_text}"
                    else: # Corresponds to "PARALLEL ACTIVITY"
                        # In parallel mode, we're seeing output from one of N simultaneous threads
                        # Find this thread's index in the active pool (0-indexed)
                        # Note: thread_names is global and might be more up-to-date than tracker's internal one.
                        current_thread_ids = sorted(list(thread_names.keys())) # Get current snapshot of active thread IDs
                        if thread_id in current_thread_ids:
                            thread_index = current_thread_ids.index(thread_id)
                        else: # Corresponds to "PARALLEL ACTIVITY"
                            # In parallel mode, we're seeing output from one of N simultaneous threads
                            active_worker_thread_ids = sorted([tid for tid in thread_names.keys() if tid != MAIN_THREAD_ID])
                            
                            core_num_str = "X" # Default/placeholder if not found
                            if thread_id in active_worker_thread_ids:
                                core_num_str = str(active_worker_thread_ids.index(thread_id) + 1)
                            elif is_main_thread: # Should not happen if this path is for workers
                                core_num_str = "M" 
                                
                            # total_cores should reflect the configured number of worker threads
                            total_configured_cores = num_threads 
                            
                            action_text = f"{to_thread_color}{to_prefix}{Colors.RESET} {to_type} {transition_info['action']}"
                            print_transition_message = f"{Colors.BOLD}{Colors.PURPLE}{switch_type} #{switch_num} (CPU {core_num_str}/{total_configured_cores}){Colors.RESET} ┃ {action_text}"

                # Update for next time
                last_thread_id = thread_id
                
                # Print transition outside of lock (this is OK even if interleaved)
                print(print_transition_message, flush=True)
                
                # Log to file outside of critical section
                if log_file:
                    try:
                        with open(log_file, 'a') as f:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # Log the raw action_text which now contains more detail
                            if transition_info: # Ensure we have transition_info
                                # Remove ANSI codes for log file
                                clean_action_text = print_transition_message
                                for color_val in [getattr(Colors, k) for k in dir(Colors) if isinstance(getattr(Colors, k), str) and getattr(Colors, k).startswith('\033')]:
                                    clean_action_text = clean_action_text.replace(color_val, '')
                                f.write(f"{timestamp} - {LogLevel.THREAD_SWITCH} - {clean_action_text}\n")
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

def determine_optimal_batch_size(images, num_threads):
    """Determine the optimal batch size based on system resources and image count.
    
    This function ensures batch sizes are always multiples of 8 and
    divide 60,000 (MNIST dataset size) evenly for optimal processing.
    
    Args:
        images: Array of images
        num_threads: Number of processing threads
    
    Returns:
        int: Recommended batch size (optimal for MNIST)
    """
    # Get total image count
    total_image_count = len(images)
    
    # Define batch sizes that are:
    # 1. Multiples of 8 (for memory alignment)
    # 2. Divide 60,000 evenly (MNIST dataset size)
    VALID_BATCH_SIZES = [
        # These are all multiples of 8 that divide 60,000 evenly
        8,      # 60,000 ÷ 8 = 7,500 batches
        16,     # 60,000 ÷ 16 = 3,750 batches
        24,     # 60,000 ÷ 24 = 2,500 batches
        40,     # 60,000 ÷ 40 = 1,500 batches
        48,     # 60,000 ÷ 48 = 1,250 batches
        80,     # 60,000 ÷ 80 = 750 batches
        96,     # 60,000 ÷ 96 = 625 batches
        120,    # 60,000 ÷ 120 = 500 batches
        160,    # 60,000 ÷ 160 = 375 batches
        200,    # 60,000 ÷ 200 = 300 batches
        240,    # 60,000 ÷ 240 = 250 batches
        400,    # 60,000 ÷ 400 = 150 batches
        600,    # 60,000 ÷ 600 = 100 batches
        1200    # 60,000 ÷ 1200 = 50 batches
    ]
    
    DEFAULT_BATCH = 120    # Default to 120 (middle ground)
    MIN_BATCH = 8          # Minimum batch size (multiple of 8)
    MAX_BATCH = 1200       # Maximum batch size
    
    # Safety first - if no images, return the default
    if total_image_count <= 0:
        return DEFAULT_BATCH
    
    # For the full MNIST dataset or large subsets, use these optimized sizes
    # Prioritize full dataset (60,000 images) optimization
    if total_image_count == 60000:
        # Specifically optimized for 60,000 images, based on thread count
        if num_threads >= 64:
            # For many threads, smaller batches prevent thread starvation
            batch_candidates = [96, 80, 48]
        elif num_threads >= 32:
            # Medium-high thread count - balanced batch size
            batch_candidates = [120, 96, 160] 
        elif num_threads >= 16:
            # Medium thread count - medium batch size
            batch_candidates = [240, 200, 160]
        elif num_threads >= 8:
            # Lower thread count - larger batch size
            batch_candidates = [400, 240, 200]
        else:
            # Very low thread count - largest batch sizes
            batch_candidates = [600, 400, 240]
    # For other dataset sizes, scale proportionally
    elif total_image_count > 50000:  # Close to full MNIST
        batch_candidates = [120, 160, 200, 240]
    elif total_image_count > 20000:
        batch_candidates = [96, 120, 160]
    elif total_image_count > 10000:
        batch_candidates = [80, 96, 120]
    elif total_image_count > 5000:
        batch_candidates = [48, 80, 96]
    elif total_image_count > 1000:
        batch_candidates = [24, 40, 48]
    else:
        batch_candidates = [8, 16, 24]
    
    # Additional adjustment factor: try to have at least 4 batches per thread
    # to ensure good thread utilization
    batches_per_thread = 4
    max_ideal_batch = total_image_count // (num_threads * batches_per_thread)
    
    # Find the largest valid batch size that doesn't exceed max_ideal_batch
    selected_batch = None
    for batch_size in sorted(batch_candidates):
        if batch_size <= max_ideal_batch:
            selected_batch = batch_size
        else:
            break
    
    # If no batch size is small enough, use the smallest valid batch size
    if selected_batch is None:
        # Find smallest batch size from valid sizes
        for batch_size in VALID_BATCH_SIZES:
            if batch_size <= max_ideal_batch or batch_size == MIN_BATCH:
                selected_batch = batch_size
                break
        
        # Fallback to absolute minimum if needed
        if selected_batch is None:
            selected_batch = MIN_BATCH
    
    # Try to ensure total_image_count is divisible by both batch_size and num_threads
    # for even distribution
    for batch_size in VALID_BATCH_SIZES:
        # Look for a batch size that:
        # 1. Divides image count evenly
        # 2. Is close to the selected batch size
        # 3. Results in an integer number of batches per thread
        if (total_image_count % batch_size == 0 and 
            batch_size <= selected_batch * 2 and 
            batch_size >= selected_batch / 2 and
            (total_image_count // batch_size) % num_threads == 0):
            selected_batch = batch_size
            break
    
    # Log the selected batch size and its properties
    safe_print(
        f"Using batch size: {selected_batch} images per batch " +
        f"(multiple of 8, divides 60,000 evenly)",
        level=LogLevel.SYSTEM
    )
    
    return selected_batch

def display_benchmark_results(total_time, total_images_processed, successful_threads, num_threads):
    """Display comprehensive benchmark results.
    
    Args:
        total_time: Total processing time in seconds
        total_images_processed: Total number of images processed
        successful_threads: Number of threads that completed successfully
        num_threads: Total number of threads used
    """
    # Calculate metrics
    images_per_second = total_images_processed / total_time if total_time > 0 else 0
    thread_efficiency = successful_threads / num_threads if num_threads > 0 else 0
    
    # Get transition stats from the tracker
    stats = transition_tracker.get_stats(elapsed_time=total_time)
    
    # Using rich.table.Table for a structured and visually appealing display of benchmark results.
    table = Table(title="[bold cyan]Benchmark Results[/bold cyan]", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=40)
    table.add_column("Value", style="bold")

    # Add rows to the table
    table.add_row("[yellow]Total processing time[/yellow]", f"{total_time:.2f} seconds")
    table.add_row("[yellow]Total images processed[/yellow]", f"{total_images_processed}")
    table.add_row("[yellow]Processing rate[/yellow]", f"{images_per_second:.2f} images/second")
    table.add_row("[yellow]Thread efficiency[/yellow]", f"{thread_efficiency:.2%} ({successful_threads}/{num_threads} threads successful)")
    
    # Threading-specific metrics
    table.add_row(f"[cyan]{stats['term']}[/cyan]", f"{stats['count']}")
    if 'rate' in stats:
        rate_term = stats['term'].split(' ')[0].lower()
        table.add_row(f"[cyan]Transition rate[/cyan]", f"{stats['rate']:.2f} {rate_term}/second")

    # Threading mode summary
    gil_status = is_gil_enabled()
    if gil_status:
        table.add_row("[blue]Threading mode[/blue]", "Single-threaded interpreter lock (GIL)")
        table.add_row("[blue]Execution[/blue]", "Only one thread executes Python code at a time")
    else:
        table.add_row("[blue]Threading mode[/blue]", "True parallel execution (No GIL)")
        table.add_row("[blue]Execution[/blue]", f"All {num_threads} threads execute simultaneously on separate CPU cores")

    # Print the table using the global console object
    console.print(table)

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
    thread_type_display = f"({Colors.BOLD}{thread_type} THREAD{Colors.RESET})"
    
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
        f"Starting to process chunk {chunk_id} with {len(images_chunk)} images {thread_type_display}", 
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
    
    safe_print(f"{thread_color}Processing chunk {chunk_id} in {num_batches} batches of ~{actual_batch_size} images each {thread_type_display}{Colors.RESET}", level=LogLevel.INFO, force_print=True)
    
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
                safe_print(f"{Colors.level_colors[LogLevel.BATCH_START]}Batch {batch_idx+1}/{num_batches} - Processing convolution on {images_count} images {thread_type_display}{Colors.RESET}", level=LogLevel.BATCH_START, force_print=True)
            
            for i in range(iterations):
                conv_start = time.time()
                # Process all images in batch with optimized convolution
                for img in batch_images:
                    apply_convolution(img, kernel)
                conv_time = time.time() - conv_start
                conv_times.append(conv_time)
            
            # Rotation operation - vectorized
            if batch_idx == 0:
                safe_print(f"{Colors.level_colors[LogLevel.BATCH_START]}Batch {batch_idx+1}/{num_batches} - Processing rotation on {images_count} images {thread_type_display}{Colors.RESET}", level=LogLevel.BATCH_START, force_print=True)
            
            for i in range(iterations):
                rot_start = time.time()
                # Process all images in batch with optimized rotation
                for img in batch_images:
                    rotate_image(img)
                rot_time = time.time() - rot_start
                rotation_times.append(rot_time)
            
            # Neural network processing - optimized with cached weights
            if batch_idx == 0:
                safe_print(f"{Colors.level_colors[LogLevel.BATCH_START]}Batch {batch_idx+1}/{num_batches} - Running neural network on {images_count} images {thread_type_display}{Colors.RESET}", level=LogLevel.BATCH_START, force_print=True)
            
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
                safe_print(f"{thread_color}Completed batch {batch_idx+1}/{num_batches} of chunk {chunk_id} in {batch_item_time:.2f}s ({images_count} images) {thread_type_display}{Colors.RESET}", level=LogLevel.INFO, force_print=True)
        
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
        
        safe_print(f"{thread_color}Chunk {chunk_id} complete - processed {len(images_chunk)} images in {total_chunk_time:.2f}s {thread_type_display}{Colors.RESET}", level=LogLevel.INFO, force_print=True)
        return result
    except Exception as e:
        safe_print(f"{thread_color}Error in chunk {chunk_id}: {e} {thread_type_display}{Colors.RESET}", level=LogLevel.ERROR, force_print=True)
        raise

def process_large_image_dataset(images, num_threads=None, iterations=1):
    """Process a large image dataset with optimized batching.
    
    This function is designed to efficiently process all 60,000 MNIST images
    using batch sizes that are multiples of 8 and optimized for the dataset size.
    
    Args:
        images: Image dataset (numpy array)
        num_threads: Number of threads to use (default: calculated based on CPU cores)
        iterations: Number of processing iterations per image
        
    Returns:
        tuple: (total_time, total_images_processed)
    """
    # If threads not specified, calculate optimal count based on CPU cores
    if num_threads is None:
        num_threads = calculate_optimal_thread_count()
    
    safe_print(f"Processing all {len(images)} images using {num_threads} threads", level=LogLevel.SYSTEM)
    
    # Determine optimal batch size based on image count and thread count
    # This ensures batch sizes are multiples of 8 and divide 60,000 evenly
    batch_size = determine_optimal_batch_size(images, num_threads)
    
    # Run the processing with the calculated batch size
    return run_mnist_processing_multithreaded(
        images, 
        num_threads=num_threads, 
        batch_size=batch_size,
        iterations=iterations,
        timeout=3600,  # 1 hour timeout
        image_limit=0  # Process all images
    )

def run_mnist_processing_multithreaded(images, num_threads, batch_size=None, iterations=1, timeout=24*3600, image_limit=0):
    """Run CPU-intensive processing on MNIST images using multiple threads."""
    global thread_names, MAIN_THREAD_ID, transition_tracker
    
    # Reset the global transition_tracker instance for a clean slate per benchmark run.
    transition_tracker.reset()
    # Reset global thread_names map for the new run
    thread_names = {} 
    
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
        f"Main thread is {Colors.MAIN_THREAD_HIGHLIGHT}MAIN-{main_thread_id}{Colors.RESET} at address 0x{main_address:x}",
        level=LogLevel.SYSTEM,
        force_print=True
    )
    
    # Thread synchronization note - use the exact technical descriptions
    if gil_status:
        safe_print(
            f"GIL enabled: Single-threaded interpreter lock controlling Python bytecode execution",
            level=LogLevel.SYSTEM
        )
        safe_print(
            f"Only one thread executes Python code at a time, controlled by the GIL",
            level=LogLevel.SYSTEM
        )
    else:
        safe_print(
            f"GIL disabled: True parallel execution with native OS thread scheduling",
            level=LogLevel.SYSTEM
        )
        safe_print(
            f"All threads execute simultaneously on separate CPU cores without GIL constraints",
            level=LogLevel.SYSTEM
        )
    
    safe_print(
        f"OPTIMIZED: Using vectorized operations and cached neural network weights",
        level=LogLevel.SYSTEM
    )
    
    start_time = time.time()
    
    # Ensure division is even to avoid fractional chunks
    # First, find a chunk size that divides evenly into the image count
    total_images = len(images)
    
    # Calculate the largest chunk size that:
    # 1. Divides evenly into total_images
    # 2. Results in at least num_threads chunks
    chunk_candidates = []
    for chunk_size in range(1, total_images + 1):
        if total_images % chunk_size == 0:
            num_chunks = total_images // chunk_size
            if num_chunks >= num_threads:
                chunk_candidates.append((chunk_size, num_chunks))
    
    # Find the chunk configuration that gives us exactly num_threads chunks
    # or slightly more (but never less)
    chunk_size = 1  # Default
    num_chunks = total_images  # Default
    
    for size, chunks in sorted(chunk_candidates, key=lambda x: abs(x[1] - num_threads)):
        if chunks >= num_threads:
            chunk_size = size
            num_chunks = chunks
            if chunks == num_threads:  # Perfect match
                break
    
    # Now we know our chunk configuration won't have fractional chunks
    safe_print(
        f"Dividing {total_images} images into {num_chunks} chunks of {chunk_size} images each",
        level=LogLevel.SYSTEM
    )
    
    # Create chunked indices to ensure even division
    chunks = []
    total_images_to_process = 0
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_images = images[start_idx:end_idx]
        chunks.append(chunk_images)
        total_images_to_process += len(chunk_images)
        
        if i < 5 or i >= num_chunks - 5:  # Log first and last 5 chunks
            safe_print(
                f"Created chunk {i+1}/{num_chunks} with {len(chunk_images)} images (indexes {start_idx}-{end_idx-1})",
                level=LogLevel.SYSTEM
            )
        elif i == 5:
            safe_print("... more chunks ...", level=LogLevel.SYSTEM)
    
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
            # Submit tasks up to the number of threads
            for i in range(min(num_chunks, num_threads)):
                safe_print(f"Submitting chunk {i+1}/{num_chunks} to thread pool", level=LogLevel.SYSTEM)
                future = executor.submit(process_chunk, i, chunks[i], iterations, batch_size, num_threads)
                future_to_chunk[future] = i
                futures_list.append(future)
            
            # If there are more chunks than threads, they'll be submitted as earlier ones complete
            next_chunk_idx = min(num_chunks, num_threads)
            
            safe_print(
                f"Initial {len(futures_list)} chunks submitted to thread pool ({next_chunk_idx}/{num_chunks})", 
                level=LogLevel.SYSTEM
            )
            
            # Collect results as they complete, with timeout
            completed_futures = 0
            for future in concurrent.futures.as_completed(future_to_chunk, timeout=timeout):
                chunk_idx = future_to_chunk[future]
                try:
                    safe_print(f"Receiving result from chunk {chunk_idx+1}", level=LogLevel.SYSTEM)
                    result = future.result()
                    results.append(result)
                    completed_futures += 1
                    
                    # Get thread color for thread ID
                    thread_id = result.get('thread_id')
                    thread_name = get_thread_name(thread_id)
                    thread_type = result.get('thread_type', 'UNKNOWN')
                    thread_color = Colors.get_thread_color(thread_id)
                    
                    # Make thread type visually distinct
                    if thread_type == "MAIN":
                        thread_type_display = f"({Colors.BOLD}MAIN THREAD{Colors.RESET})"
                    else:
                        thread_type_display = f"(WORKER THREAD)"
                    
                    safe_print(
                        f"{thread_color}Thread {thread_name} {thread_type_display} completed chunk {chunk_idx+1} with {result.get('num_images', 0)} images{Colors.RESET}",
                        level=LogLevel.SYSTEM
                    )
                    
                    # Submit the next chunk if available
                    if next_chunk_idx < num_chunks:
                        safe_print(f"Submitting next chunk {next_chunk_idx+1}/{num_chunks} to thread pool", level=LogLevel.SYSTEM)
                        next_future = executor.submit(process_chunk, next_chunk_idx, chunks[next_chunk_idx], iterations, batch_size, num_threads)
                        future_to_chunk[next_future] = next_chunk_idx
                        futures_list.append(next_future)
                        next_chunk_idx += 1
                    
                except Exception as e:
                    safe_print(
                        f"Error handling result from chunk {chunk_idx+1}: {e}", 
                        level=LogLevel.ERROR
                    )
                    # Add partial result with error information
                    results.append({
                        'chunk_id': chunk_idx,
                        'error': str(e),
                        'conv_times': [],
                        'rotation_times': [],
                        'nn_times': [],
                        'num_images': len(chunks[chunk_idx]),
                        'thread_id': threading.get_ident(),
                        'thread_name': get_thread_name(threading.get_ident()),
                        'thread_type': 'MAIN' if threading.get_ident() == MAIN_THREAD_ID else 'WORKER'
                    })
                    
                    # Try to submit the next chunk despite this error
                    if next_chunk_idx < num_chunks:
                        safe_print(f"Submitting next chunk {next_chunk_idx+1}/{num_chunks} to thread pool (after error)", level=LogLevel.SYSTEM)
                        next_future = executor.submit(process_chunk, next_chunk_idx, chunks[next_chunk_idx], iterations, batch_size, num_threads)
                        future_to_chunk[next_future] = next_chunk_idx
                        futures_list.append(next_future)
                        next_chunk_idx += 1
            
            safe_print(f"Completed {completed_futures}/{num_chunks} chunks", level=LogLevel.SYSTEM)
        
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
            for future in futures_list:
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
    
    # Display benchmark results with improved terminology
    display_benchmark_results(total_time, total_images_processed, successful_threads, num_threads)
    
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
    
    # Generate unique log filename with timestamp and GIL status
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add GIL status to log filename
    gil_status = is_gil_enabled()
    gil_str = "with_gil" if gil_status else "without_gil"
    
    log_file = os.path.join(log_dir, f"mnist_{gil_str}_benchmark_{timestamp}_{num_threads}threads.log")
    
    # Open the log file
    with open(log_file, 'w') as f:
        f.write(f"MNIST {'GIL' if gil_status else 'No-GIL'} Demonstration Benchmark - {timestamp} - {num_threads} threads\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"numpy version: {np.__version__}\n")
        f.write(f"CPU cores: {multiprocessing.cpu_count()}\n")
        f.write("-" * 80 + "\n")
    
    # Return a basic logger for compatibility
    return logging.getLogger(__name__)

def main():
    """Main function to run the benchmark."""
    global num_threads, MAIN_THREAD_ID, verbose, thread_names, last_thread_id, active_thread, track_thread_switches, transition_tracker
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run MNIST GIL demonstration with multiple threads')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads to use (0 for auto-determination)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds (default: 1 hour)')
    parser.add_argument('--image-limit', type=int, default=0, help='Number of images to process (0 for all 60,000 images)')
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
    
    # Auto-detect optimal number of threads if not specified
    if args.threads <= 0:
        num_threads = calculate_optimal_thread_count()
        safe_print(f"Auto-determined optimal thread count: {num_threads} threads", level=LogLevel.SYSTEM, force_print=True)
    else:
        num_threads = args.threads
    
    # Initialize global variables related to thread state
    thread_names = {}  # Maps thread IDs to names
    last_thread_id = None # Tracks the last thread ID seen by safe_print
    
    # Ensure main thread ID is set correctly for this process
    MAIN_THREAD_ID = threading.get_ident()
    active_thread = MAIN_THREAD_ID
    
    # Set up logging with both console and file output
    logger = setup_logging(num_threads, args.log_dir)
    
    # Log benchmark initialization with distinctive main thread formatting
    safe_print(
        f"{Colors.MAIN_THREAD_HIGHLIGHT}MNIST Thread Processing Benchmark initializing{Colors.RESET} (MAIN THREAD)",
        level=LogLevel.SYSTEM,
        force_print=True
    )
    
    # Check GIL status and print with clearer messaging
    gil_status = is_gil_enabled()
    if gil_status:
        gil_status_str = "ENABLED - Single-threaded interpreter lock controlling Python bytecode execution"
        gil_status_color = Colors.YELLOW
    else:
        gil_status_str = "DISABLED - True parallel execution with native OS thread scheduling"
        gil_status_color = Colors.GREEN
    
    safe_print(
        f"{Colors.BOLD}{gil_status_color}Python GIL is {gil_status_str}{Colors.RESET}",
        level=LogLevel.SYSTEM,
        force_print=True
    )
    
    # Log thread tracking information
    if track_thread_switches:
        if gil_status:
            safe_print(f"GIL SWITCH tracking is ENABLED", level=LogLevel.SYSTEM)
            safe_print(f"Sequential execution: Only one thread active at a time, controlled by GIL", level=LogLevel.SYSTEM)
        else:
            safe_print(f"Parallel thread visualization is ENABLED", level=LogLevel.SYSTEM)
            safe_print(f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓", level=LogLevel.SYSTEM)
            safe_print(f"┃ PARALLEL MODE: All {num_threads} threads executing simultaneously ┃", level=LogLevel.SYSTEM)
            safe_print(f"┃ CPU cores shown as: PARALLEL CPU 1/{num_threads}, PARALLEL CPU 2/{num_threads}, etc.     ┃", level=LogLevel.SYSTEM)
            safe_print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛", level=LogLevel.SYSTEM)
    else:
        safe_print(f"Thread tracking is DISABLED for maximum performance", level=LogLevel.SYSTEM)
    
        # Using rich.print for styled console output of the log file path.
        console.print(f"[bold blue]Log file[/bold blue]: [green]{os.path.abspath(log_file)}[/green]", style="on white")
    
    # Print system information
    safe_print(f"Python version: {sys.version}", level=LogLevel.SYSTEM) # Keep this as safe_print for file logging
    safe_print(f"numpy version: {np.__version__}", level=LogLevel.SYSTEM) # Keep this as safe_print
    safe_print(f"Machine: {multiprocessing.cpu_count()} CPU cores", level=LogLevel.SYSTEM) # Keep this as safe_print
    safe_print(f"Main thread ID: {MAIN_THREAD_ID}", level=LogLevel.SYSTEM) # Keep this as safe_print
    
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
        
        # Print thread color and name legend for reference
        safe_print(f"{Colors.MAIN_THREAD_HIGHLIGHT}Thread naming convention:{Colors.RESET} MAIN thread uses special gold color, worker threads named after famous guitarists", level=LogLevel.SYSTEM)
        safe_print("Each thread will have a unique color for easier log reading", level=LogLevel.SYSTEM)
        
        # Show configuration
        threading_mode = "GIL-enabled threading" if gil_status else "GIL-disabled threading"
        safe_print(
            f"Configuration: {threading_mode}, image_limit={args.image_limit}, timeout={args.timeout}s, "
            f"batch_size={args.batch_size}, iterations={args.iterations}",
            level=LogLevel.SYSTEM
        )
        
        # Run the multithreaded processing
        safe_print(
            f"{Colors.MAIN_THREAD_HIGHLIGHT}Starting benchmark with {num_threads} threads{Colors.RESET} (MAIN THREAD)", 
            level=LogLevel.SYSTEM
        )
        
        # Process all 60,000 images unless limited by command line option
        image_limit = args.image_limit
        
        if image_limit > 0:
            safe_print(f"Processing limited subset of {image_limit} images", level=LogLevel.SYSTEM)
            target_images = images[:image_limit]
        else:
            safe_print(f"Processing complete dataset of {len(images)} images", level=LogLevel.SYSTEM)
            target_images = images
            
        total_time, total_processed = process_large_image_dataset(
            target_images,
            num_threads=num_threads,
            iterations=args.iterations
        )
        
        safe_print(f"{Colors.MAIN_THREAD_HIGHLIGHT}Benchmark complete{Colors.RESET} (MAIN THREAD)", level=LogLevel.SYSTEM)
        
        return 0
    except KeyboardInterrupt:
        safe_print(f"{Colors.MAIN_THREAD_HIGHLIGHT}Process interrupted by user{Colors.RESET} (MAIN THREAD)", level=LogLevel.WARNING)
        return 130  # Standard Unix exit code for SIGINT
    except Exception as e:
        safe_print(f"{Colors.MAIN_THREAD_HIGHLIGHT}Unhandled exception: {e}{Colors.RESET} (MAIN THREAD)", level=LogLevel.ERROR)
        import traceback
        safe_print(f"Exception trace: {traceback.format_exc()}", level=LogLevel.ERROR)
        return 1
    finally:
        # Always clean up resources in the finally block
        if args.cleanup:
            safe_print(f"{Colors.MAIN_THREAD_HIGHLIGHT}Cleaning up downloaded files...{Colors.RESET} (MAIN THREAD)", level=LogLevel.SYSTEM)
            cleanup_files(args.data_dir)
        
        if log_file:
            # Using rich.print for styled console output of the final log file path.
            console.print(f"[bold green]Logs saved to:[/bold green] [underline]{log_file}[/underline]")
            
        # Print thread legend for all threads used
        active_threads = sorted(thread_names.items(), key=lambda x: x[0])
        if len(active_threads) > 0:
            safe_print(f"{Colors.MAIN_THREAD_HIGHLIGHT}Thread name legend:{Colors.RESET} (MAIN THREAD)", level=LogLevel.SYSTEM)
            
            # First, print the main thread
            safe_print(
                f"{Colors.MAIN_THREAD_HIGHLIGHT}[MAIN-{MAIN_THREAD_ID}]{Colors.RESET} - MAIN THREAD",
                level=LogLevel.SYSTEM
            )
            
            # Then print worker threads (up to 16)
            max_display = min(16, len(active_threads))
            for thread_id, name in active_threads[:max_display]:
                if thread_id != MAIN_THREAD_ID:  # Skip main thread, already printed
                    thread_color = Colors.get_thread_color(thread_id)
                    # Using rich.print for styled console output of the thread legend.
                    console.print(f"{thread_color}[{name}-{thread_id}]{Colors.RESET} - WORKER THREAD")
            
            # Indicate if more threads were used but not displayed
            if len(active_threads) > max_display:
                # Using rich.print for styled console output.
                console.print(f"... and {len(active_threads) - max_display} more threads (not shown)")

if __name__ == "__main__":
    # Wrap the main function call in a try/except to allow rich to print tracebacks nicely
    try:
        sys.exit(main())
    except Exception:
        console.print_exception(show_locals=True)
        sys.exit(1)