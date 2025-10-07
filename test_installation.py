#!/usr/bin/env python3
"""
Simple installation test for FaceSwap Application.
This script verifies that all dependencies are properly installed.
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'PIL',
        'mtcnn',
        'albumentations',
        'tqdm',
        'scipy',
        'sklearn'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_faceswap_modules():
    """Test if FaceSwap modules can be imported."""
    faceswap_modules = [
        'faceswap.processing.face_detector',
        'faceswap.processing.face_swapper',
        'faceswap.processing.video_processor',
        'faceswap.processing.data_processor',
        'faceswap.models.trainer',
        'faceswap.utils.logger',
        'faceswap.utils.progress_tracker',
        'faceswap.utils.file_manager',
        'faceswap.utils.gpu_manager',
        'faceswap.utils.error_handler'
    ]
    
    failed_imports = []
    
    for module in faceswap_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} - OK")
        except ImportError as e:
            print(f"❌ {module} - FAILED: {e}")
            failed_imports.append(module)
    
    return failed_imports

def test_sample_data():
    """Test if sample data directories exist."""
    sample_paths = [
        Path('sample_data/video'),
        Path('sample_data/images')
    ]
    
    missing_dirs = []
    
    for path in sample_paths:
        if path.exists() and path.is_dir():
            file_count = len([f for f in path.glob('*') if f.is_file() and not f.name.endswith('.md')])
            print(f"✅ {path} - Directory exists ({file_count} data files)")
        else:
            print(f"❌ {path} - Missing or not a directory")
            missing_dirs.append(str(path))
    
    return missing_dirs

def main():
    """Run all installation tests."""
    print("🧪 FaceSwap Application Installation Test")
    print("=" * 50)
    
    print("\n📦 Testing Python package imports...")
    failed_packages = test_imports()
    
    print("\n🔧 Testing FaceSwap modules...")
    failed_modules = test_faceswap_modules()
    
    print("\n📁 Testing sample data...")
    missing_dirs = test_sample_data()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    if not failed_packages and not failed_modules and not missing_dirs:
        print("🎉 All tests passed! Installation is successful.")
        print("\n🚀 You can now run:")
        print("   python faceswap.py sample_data/video/sample.mp4 sample_data/images/")
        return 0
    else:
        print("❌ Some tests failed:")
        if failed_packages:
            print(f"   - Failed package imports: {', '.join(failed_packages)}")
        if failed_modules:
            print(f"   - Failed module imports: {', '.join(failed_modules)}")
        if missing_dirs:
            print(f"   - Missing directories: {', '.join(missing_dirs)}")
        
        print("\n🔧 To fix issues:")
        print("   1. Run: pip install -r requirements.txt")
        print("   2. Add sample data to sample_data/video/ and sample_data/images/")
        print("   3. Check Python version (3.8+ required)")
        return 1

if __name__ == "__main__":
    sys.exit(main())