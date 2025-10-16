#!/usr/bin/env python3
"""
Dataset Download Script for Online News Popularity Analysis

This script helps download the required dataset for the project.
"""

import os
import requests
import zipfile
from pathlib import Path

def download_dataset():
    """Download the Online News Popularity dataset."""
    
    # Dataset URL from UCI ML Repository
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
    
    # Create data directory
    data_dir = Path("data/OnlineNewsPopularity")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists
    csv_file = data_dir / "OnlineNewsPopularity.csv"
    if csv_file.exists():
        print(f"✅ Dataset already exists at {csv_file}")
        return
    
    print("📥 Downloading Online News Popularity dataset...")
    print(f"URL: {dataset_url}")
    
    try:
        # Download the zip file
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        zip_file = data_dir / "OnlineNewsPopularity.zip"
        
        # Save zip file
        with open(zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded: {zip_file}")
        
        # Extract the zip file
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove zip file
        zip_file.unlink()
        
        print(f"✅ Dataset extracted to: {data_dir}")
        print(f"📊 Dataset file: {csv_file}")
        
        # Verify the file exists
        if csv_file.exists():
            file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
            print(f"📏 File size: {file_size:.2f} MB")
            print("🎉 Dataset ready for use!")
        else:
            print("❌ Error: Dataset file not found after extraction")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Please download manually from:")
        print("   https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity")
    except zipfile.BadZipFile as e:
        print(f"❌ Error extracting dataset: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function."""
    print("=" * 60)
    print("ONLINE NEWS POPULARITY DATASET DOWNLOADER")
    print("=" * 60)
    
    download_dataset()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Run: python main.py")
    print("2. Or: streamlit run dashboard.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
