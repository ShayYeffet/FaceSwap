# Sample Data for FaceSwap Application

This directory contains sample data for testing and demonstrating the FaceSwap application.

## Directory Structure

```
sample_data/
├── README.md                 # This file
├── demo_video.mp4           # Sample input video (placeholder)
├── face_dataset/            # Sample face images for training
│   ├── face_001.jpg         # Sample face image 1
│   ├── face_002.jpg         # Sample face image 2
│   ├── face_003.jpg         # Sample face image 3
│   ├── ...                  # Additional face images
│   └── README.md            # Dataset information
├── expected_output/         # Expected output examples
│   ├── demo_output.mp4      # Expected result video
│   └── README.md            # Output information
└── demo_script.py           # Demo script showing typical usage

```

## Usage Instructions

### Basic Demo
```bash
# Run the basic demo
python demo_script.py

# Or run manually with sample data
python faceswap.py sample_data/demo_video.mp4 sample_data/face_dataset/
```

### Advanced Demo
```bash
# Run with custom parameters
python faceswap.py sample_data/demo_video.mp4 sample_data/face_dataset/ \
    --output my_result.mp4 \
    --epochs 50 \
    --batch-size 8 \
    --quality high
```

## Sample Data Information

### Demo Video (`demo_video.mp4`)
- **Duration**: ~10 seconds
- **Resolution**: 720p (1280x720)
- **Frame Rate**: 30 FPS
- **Content**: Person speaking to camera with clear facial features
- **Format**: MP4 (H.264)

### Face Dataset (`face_dataset/`)
- **Image Count**: 20+ high-quality face images
- **Resolution**: Various (minimum 256x256)
- **Format**: JPG/PNG
- **Content**: Clear, front-facing photos with good lighting
- **Diversity**: Various expressions and slight pose variations

### Expected Output
- **Processing Time**: ~2-5 minutes on GPU, 10-30 minutes on CPU
- **Quality**: Demonstrates successful face replacement with natural blending
- **File Size**: Similar to input video size

## Requirements for Your Own Data

### Video Requirements
- **Formats**: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
- **Resolution**: Minimum 480p, recommended 720p or higher
- **Duration**: Any length (longer videos take more time)
- **Content**: Clear facial features, good lighting, minimal motion blur

### Dataset Requirements
- **Minimum Images**: 10 (recommended 50-100 for best results)
- **Image Quality**: High resolution, clear facial features
- **Lighting**: Well-lit, consistent lighting conditions
- **Pose**: Primarily front-facing, some variation acceptable
- **Expression**: Various expressions help model generalization

## Troubleshooting

### Common Issues
1. **"No faces detected"**: Ensure dataset images have clear, visible faces
2. **Poor quality results**: Use more training images and increase epochs
3. **GPU memory errors**: Reduce batch size or use CPU processing
4. **Long processing time**: Normal for CPU processing, consider GPU upgrade

### Performance Tips
- Use GPU for significantly faster processing
- More training images generally improve quality
- Higher resolution faces (512px) give better results but require more memory
- Balance training epochs vs. processing time based on your needs

## Creating Your Own Sample Data

To create your own sample data:

1. **Collect face images**: 50-100 high-quality photos of the target person
2. **Prepare video**: Choose a video with clear facial features and good lighting
3. **Organize files**: Place images in a folder, ensure video is in supported format
4. **Test run**: Start with fewer epochs (20-50) to test before full processing

## Legal and Ethical Considerations

⚠️ **Important**: This tool is for educational and research purposes. Always:
- Obtain proper consent before using someone's likeness
- Respect privacy and intellectual property rights
- Use responsibly and ethically
- Be aware of local laws regarding deepfake technology
- Consider the potential impact of generated content

## Support

For issues with sample data or the application:
1. Check the main README.md for installation instructions
2. Verify your system meets the requirements
3. Review the troubleshooting section above
4. Check log files for detailed error information