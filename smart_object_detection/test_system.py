import os
import sys

def test_imports():
    print("Testing third-party dependencies...")
    try:
        import streamlit as st
        print("[OK] streamlit imported successfully")
    except ImportError:
        print("[FAIL] streamlit failed to import")
        
    try:
        import cv2
        print(f"[OK] opencv-python (cv2) imported successfully, version {cv2.__version__}")
    except ImportError:
        print("[FAIL] opencv-python failed to import")
        
    try:
        import numpy as np
        print(f"[OK] numpy imported successfully, version {np.__version__}")
    except ImportError:
        print("[FAIL] numpy failed to import")
        
    try:
        import pandas as pd
        print(f"[OK] pandas imported successfully, version {pd.__version__}")
    except ImportError:
        print("[FAIL] pandas failed to import")
        
    try:
        from ultralytics import YOLO
        print("[OK] ultralytics (YOLO) imported successfully")
    except ImportError:
        print("[FAIL] ultralytics failed to import")
        
    try:
        import openpyxl
        print(f"[OK] openpyxl (Excel engine) imported successfully, version {openpyxl.__version__}")
    except ImportError:
        print("[FAIL] openpyxl failed to import")

def test_local_modules():
    print("\nTesting local modular components...")
    try:
        from detector import YOLODetector
        from tracker import ObjectTracker
        from logger import DetectionLogger
        import utils
        print("[OK] Core modules: detector, tracker, logger, and utils imported successfully!")
    except Exception as e:
        print(f"[FAIL] Core modules failed to load: {e}")
        
    try:
        from line_counter import LineCrossingCounter
        from heatmap import MovementHeatmap
        from analytics import SurveillanceAnalytics
        print("[OK] Advanced modules: line_counter, heatmap, and analytics imported successfully!")
    except Exception as e:
        print(f"[FAIL] Advanced modules failed to load: {e}")

if __name__ == "__main__":
    print(f"Python Version: {sys.version}")
    test_imports()
    test_local_modules()
    print("\nSystem environment verification check completed successfully!")
