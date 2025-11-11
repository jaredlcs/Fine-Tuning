#!/usr/bin/env python3
"""
Quick setup script for Prompt Injection Detection project
Installs dependencies and runs basic environment checks
"""

import subprocess
import sys
import importlib
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - Available")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not available")
        return False

def main():
    print("🚀 Prompt Injection Detection - Project Setup")
    print("=" * 50)
    
    # System info
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()
    
    # Install requirements
    print("📦 Installing dependencies...")
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements"
    )
    
    if not success:
        print("⚠️ Failed to install some packages. Trying individual installation...")
        critical_packages = [
            "transformers", "datasets", "torch", "scikit-learn", 
            "matplotlib", "seaborn", "pandas", "numpy", "tqdm"
        ]
        
        for package in critical_packages:
            run_command(
                f"{sys.executable} -m pip install {package}",
                f"Installing {package}"
            )
    
    print("\n🔍 Checking installed packages...")
    
    # Check critical packages
    packages_to_check = [
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("torch", "torch"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("evaluate", "evaluate"),
        ("accelerate", "accelerate")
    ]
    
    available_count = 0
    for package_name, import_name in packages_to_check:
        if check_package(package_name, import_name):
            available_count += 1
    
    print(f"\n📊 Package Status: {available_count}/{len(packages_to_check)} available")
    
    if available_count >= len(packages_to_check) - 2:  # Allow 2 missing non-critical packages
        print("🎉 Setup completed successfully!")
        print("\n📝 Next steps:")
        print("1. Open 'part1_training_evaluation.ipynb' to start with fine-tuning")
        print("2. Open 'part2_generalization_test.ipynb' to test on new datasets")
        print("3. Check README.md for detailed instructions")
    else:
        print("⚠️ Setup incomplete. Some packages are missing.")
        print("Please install missing packages manually or check your environment.")
    
    # Test GPU availability
    print("\n🖥️ Hardware check:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name()}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("ℹ️ GPU not available - will use CPU (slower training)")
    except:
        print("❓ Could not check GPU status")

if __name__ == "__main__":
    main()