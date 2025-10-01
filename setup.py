
import subprocess
import sys
import os

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def setup_system():
    print("Setting up Heart Disease Prediction System...")
    print("=" * 50)
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt"):
        return False
    
    print("\nCreating directories...")
    directories = ['data', 'models', 'utils', 'static']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\nTraining ML model...")
    if not run_command("python model_training.py"):
        print("⚠️ Model training failed, but you can run it manually later")
    
    print("\nTesting system...")
    if not run_command("python test_system.py"):
        print("⚠️ System test failed, but you can run it manually later")
    
    print("\n🎉 Setup completed!")
    print("\nTo start the application:")
    print("  Windows: Double-click run_app.bat")
    print("  Or run: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    setup_system()
