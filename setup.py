#!/usr/bin/env python3
"""
FaceSwap Application Setup Script

This setup script allows for easy installation of the FaceSwap application
and its dependencies across different platforms.
"""

from setuptools import setup, find_packages
import os
import sys

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

# Platform-specific requirements
def get_platform_requirements():
    """Get platform-specific requirements"""
    platform_reqs = []
    
    if sys.platform.startswith("win"):
        # Windows-specific packages
        platform_reqs.extend([
            "pywin32>=306",  # Windows API access
        ])
    elif sys.platform.startswith("darwin"):
        # macOS-specific packages
        platform_reqs.extend([
            # No specific requirements for macOS currently
        ])
    elif sys.platform.startswith("linux"):
        # Linux-specific packages
        platform_reqs.extend([
            # Most Linux dependencies are handled by system package manager
        ])
    
    return platform_reqs

# Check Python version
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required. You are using Python {}.{}.".format(
        sys.version_info.major, sys.version_info.minor))

setup(
    name="faceswap-application",
    version="1.0.0",
    author="FaceSwap Team",
    author_email="contact@faceswap.dev",
    description="A user-friendly deep learning based face swapping tool for videos",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/faceswap/faceswap-application",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements() + get_platform_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            # GPU-specific packages (handled by PyTorch installation)
        ],
        "visualization": [
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
        ],
        "monitoring": [
            "psutil>=5.9.0",
            "GPUtil>=1.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "faceswap=faceswap:main",
        ],
    },
    include_package_data=True,
    package_data={
        "faceswap": [
            "*.yaml",
            "*.yml",
            "*.json",
        ],
    },
    zip_safe=False,
    keywords="faceswap deepfake video processing machine learning pytorch",
    project_urls={
        "Bug Reports": "https://github.com/faceswap/faceswap-application/issues",
        "Source": "https://github.com/faceswap/faceswap-application",
        "Documentation": "https://github.com/faceswap/faceswap-application/blob/main/README.md",
    },
)