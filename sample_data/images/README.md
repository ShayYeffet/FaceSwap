# Sample Face Images

This directory should contain face images for training the face swap model.

## Requirements

- **Format**: JPG, PNG, or JPEG images
- **Quantity**: At least 10 images recommended (more = better quality)
- **Quality**: Clear, well-lit face images
- **Variety**: Different angles and expressions for best results

## Usage

Place your target face images in this directory, then run:

```bash
python faceswap.py sample_data/video/sample.mp4 sample_data/images/
```

## Example Structure

```
sample_data/images/
├── face1.jpg
├── face2.png
├── face3.jpg
└── ...
```

## Tips for Best Results

- Use high-resolution images (at least 256x256 pixels)
- Ensure faces are clearly visible and well-lit
- Include variety in facial expressions and angles
- Remove blurry or heavily occluded images
- Use images of the same person for consistent results