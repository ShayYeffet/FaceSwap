# FaceSwap Application

A user-friendly deep learning based face swapping tool that allows you to replace faces in videos using GPU-accelerated training. Simply provide a video file and a dataset of face images to generate realistic face-swapped videos.

## Features

- **Easy to Use**: Simple command-line interface - just run `python faceswap.py video.avi dataset_folder`
- **GPU Accelerated**: Automatic GPU detection with CUDA support for fast training and processing
- **Robust Face Detection**: Uses MTCNN for accurate face detection and alignment
- **High Quality Results**: Advanced autoencoder architecture with adversarial training
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Comprehensive Error Handling**: Clear error messages and graceful fallbacks

## Installation

### Prerequisites

- Python 3.8 or higher
- For GPU acceleration: NVIDIA GPU with CUDA support (recommended)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShayYeffet/FaceSwap.git
   cd faceswap
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python faceswap.py --help
   ```

### Platform-Specific Installation

#### Windows
```bash
# Install Python 3.8+ from python.org
# Install CUDA Toolkit 11.8+ from NVIDIA (for GPU support)
pip install -r requirements.txt
```

#### macOS
```bash
# Install Python via Homebrew
brew install python
pip install -r requirements.txt
```

#### Linux (Ubuntu/Debian)
```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip
pip3 install -r requirements.txt

# For GPU support, install CUDA toolkit
# Follow NVIDIA CUDA installation guide for your distribution
```

## Usage

### Basic Usage

Replace faces in a video using a dataset of target face images:

```bash
python faceswap.py input_video.mp4 face_dataset_folder/
```

### Advanced Usage

Customize training parameters for better results:

```bash
python faceswap.py input_video.mp4 face_dataset_folder/ \
    --epochs 150 \
    --batch-size 8 \
    --learning-rate 0.0001 \
    --output custom_output.mp4
```

### Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `video_path` | Input video file path | Required | `video.mp4` |
| `dataset_path` | Folder containing target face images | Required | `faces/` |
| `--epochs` | Number of training epochs | 100 | `--epochs 150` |
| `--batch-size` | Training batch size | 16 | `--batch-size 8` |
| `--learning-rate` | Learning rate for training | 0.0001 | `--learning-rate 0.0002` |
| `--output` | Output video filename | Auto-generated | `--output result.mp4` |
| `--device` | Force device (cpu/cuda/auto) | auto | `--device cuda` |
| `--help` | Show help message | - | `--help` |

### Dataset Requirements

- **Format**: JPG, PNG, or JPEG images
- **Quantity**: At least 10 images recommended (more = better quality)
- **Quality**: Clear, well-lit face images
- **Variety**: Different angles and expressions for best results

Example dataset structure:
```
face_dataset_folder/
├── image1.jpg
├── image2.png
├── image3.jpg
└── ...
```

## Examples

### Example 1: Basic Face Swap
```bash
python faceswap.py my_video.mp4 celebrity_faces/
```
Output: `my_video_faceswapped.mp4`

### Example 2: High Quality Training
```bash
python faceswap.py interview.avi actor_dataset/ --epochs 200 --batch-size 4
```

### Example 3: CPU-Only Processing
```bash
python faceswap.py video.mp4 faces/ --device cpu
```

## Project Structure

```
faceswap-application/
├── faceswap.py                 # Main CLI application
├── requirements.txt            # Python dependencies
├── README.md                  # This documentation
└── faceswap/                  # Core package
    ├── __init__.py
    ├── models/                # Neural network models
    │   ├── __init__.py
    │   ├── autoencoder.py     # Main model architecture
    │   ├── losses.py          # Loss functions
    │   └── trainer.py         # Training engine
    ├── processing/            # Face and video processing
    │   ├── __init__.py
    │   ├── face_detector.py   # Face detection using MTCNN
    │   ├── data_processor.py  # Dataset preprocessing
    │   ├── face_swapper.py    # Face swapping logic
    │   └── video_processor.py # Video I/O and processing
    └── utils/                 # Utility modules
        ├── __init__.py
        ├── gpu_manager.py     # GPU detection and management
        ├── file_manager.py    # File validation
        ├── logger.py          # Logging system
        ├── progress_tracker.py # Progress tracking
        └── error_handler.py   # Error handling
```

## Troubleshooting

### Common Issues

#### "No GPU detected" Warning
**Problem**: Application falls back to CPU processing
**Solutions**:
- Install NVIDIA CUDA Toolkit (11.8 or higher)
- Verify GPU compatibility: `nvidia-smi`
- Install PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

#### "No faces detected in dataset"
**Problem**: Face detection fails on dataset images
**Solutions**:
- Ensure images contain clear, visible faces
- Use well-lit, high-resolution images
- Remove blurry or heavily occluded face images
- Try different image formats (JPG, PNG)

#### "CUDA out of memory" Error
**Problem**: GPU memory exhausted during training
**Solutions**:
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Use CPU processing: `--device cpu`
- Close other GPU-intensive applications
- Use smaller input video resolution

#### "Video format not supported"
**Problem**: Input video cannot be processed
**Solutions**:
- Convert to supported format (MP4, AVI, MOV, MKV)
- Use FFmpeg to convert: `ffmpeg -i input.webm output.mp4`
- Check video file integrity

#### Slow Processing Speed
**Problem**: Training or processing takes too long
**Solutions**:
- Ensure GPU acceleration is working
- Reduce video resolution or duration for testing
- Increase batch size if GPU memory allows
- Use fewer training epochs for quick tests

#### Poor Quality Results
**Problem**: Face swap looks unrealistic
**Solutions**:
- Increase training epochs: `--epochs 200`
- Use more diverse dataset images (different angles, lighting)
- Ensure dataset faces are similar to target faces in video
- Try different learning rates: `--learning-rate 0.00005`

### System Requirements

#### Minimum Requirements
- **OS**: Windows 10, macOS 10.15, or Linux (Ubuntu 18.04+)
- **RAM**: 8GB
- **Storage**: 5GB free space
- **CPU**: Intel i5 or AMD Ryzen 5 (with AVX support)

#### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Linux (Ubuntu 20.04+)
- **RAM**: 16GB or more
- **Storage**: 10GB+ free space (for models and temporary files)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (GTX 1060 or better)
- **CPU**: Intel i7 or AMD Ryzen 7

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for detailed error messages in the console output
2. **Verify installation**: Run `python faceswap.py --help` to ensure proper setup
3. **Test with sample data**: Try with a short video and small dataset first
4. **Check system resources**: Monitor GPU/CPU usage during processing

### Performance Tips

- **Use GPU acceleration** for 10-20x speed improvement
- **Start with short videos** (30 seconds) for testing
- **Use 50-100 high-quality dataset images** for best results
- **Monitor system resources** to avoid memory issues
- **Save checkpoints** are created automatically every 10 epochs

## Technical Details

### Model Architecture
- **Type**: Autoencoder with shared encoder and separate decoders
- **Input**: 256x256 RGB face images
- **Training**: Adversarial loss with perceptual loss for identity preservation
- **Framework**: PyTorch with CUDA acceleration

### Dependencies
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision and video processing
- **MTCNN**: Face detection and alignment
- **FFmpeg**: Video encoding/decoding
- **NumPy**: Numerical computing

### Supported Formats
- **Video**: MP4, AVI, MOV, MKV
- **Images**: JPG, JPEG, PNG
- **Output**: MP4 (H.264 encoding)

## License

This project is for educational and research purposes. Please use responsibly and be aware of ethical considerations when creating deepfake content.

## Contributing

Contributions are welcome! Please ensure all code follows the existing style and includes appropriate tests.
