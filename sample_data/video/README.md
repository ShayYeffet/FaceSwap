# Sample Video Files

This directory should contain video files for face swapping.

## Requirements

- **Format**: MP4, AVI, MOV, or MKV
- **Quality**: Clear faces throughout the video
- **Lighting**: Good lighting conditions
- **Resolution**: Up to 1920x1080 for optimal performance

## Usage

Place your input video in this directory, then run:

```bash
python faceswap.py sample_data/video/your_video.mp4 sample_data/images/
```

## Example Structure

```
sample_data/video/
├── sample.mp4
├── interview.avi
└── ...
```

## Tips for Best Results

- Use videos with clear, well-lit faces
- Ensure faces are visible throughout most of the video
- Shorter videos (30 seconds to 2 minutes) process faster
- Higher resolution videos may require more processing time
- Ensure the video format is supported by OpenCV