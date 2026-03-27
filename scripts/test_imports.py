"""Test that all required libraries are installed and importable."""

import sys

def test_imports():
    results = []
    libraries = [
        ("ultralytics", "YOLOv8 framework"),
        ("albumentations", "Data augmentation"),
        ("cv2", "OpenCV image processing"),
        ("numpy", "Numerical computing"),
        ("PIL", "Pillow image I/O"),
        ("tqdm", "Progress bars"),
        ("imagehash", "Perceptual hashing"),
        ("roboflow", "Roboflow API"),
        ("dotenv", "Environment variables"),
        ("matplotlib", "Plotting"),
        ("seaborn", "Statistical plotting"),
        ("torch", "PyTorch"),
        ("torchvision", "PyTorch vision"),
        ("kaggle", "Kaggle API"),
        ("pandas", "Data analysis"),
        ("scipy", "Scientific computing"),
    ]

    print("Testing library imports...\n")
    all_ok = True
    for lib, desc in libraries:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "N/A")
            print(f"  [OK] {lib:20s} v{ver:15s}  ({desc})")
            results.append((lib, True, ver))
        except ImportError as e:
            print(f"  [FAIL] {lib:20s}  ({desc}) — {e}")
            results.append((lib, False, None))
            all_ok = False

    # Test MPS (Apple Silicon GPU)
    print("\nGPU Backend Check:")
    try:
        import torch
        if torch.backends.mps.is_available():
            print("  [OK] Apple MPS (Metal) is available")
        else:
            print("  [WARN] MPS not available, will use CPU")
        print(f"  PyTorch device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    except Exception as e:
        print(f"  [FAIL] GPU check failed: {e}")

    # Test YOLO can load
    print("\nYOLO Quick Check:")
    try:
        from ultralytics import YOLO
        print("  [OK] YOLO class importable")
    except Exception as e:
        print(f"  [FAIL] YOLO import: {e}")
        all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("All imports successful!")
    else:
        print("Some imports FAILED — check above.")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
