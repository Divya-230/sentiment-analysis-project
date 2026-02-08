import subprocess
import sys

def install_packages():
    packages = [
        'flask==2.3.3',
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'nltk==3.8.1',
        'textblob==0.17.1',
        'matplotlib==3.7.2',
        'seaborn==0.12.2',
        'plotly==5.15.0',
        'wordcloud==1.9.2',
        'joblib==1.3.2',
        'openpyxl==3.1.2'  # For Excel support
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\nAll packages installed! Now downloading NLTK data...")
    
    # Download NLTK data
    import nltk
    nltk.download('punkt', quiet=False)
    nltk.download('stopwords', quiet=False)
    nltk.download('wordnet', quiet=False)
    nltk.download('omw-1.4', quiet=False)
    
    print("✓ NLTK data downloaded successfully")
    print("\nYou can now run the application with: python app.py")

if __name__ == "__main__":
    install_packages()