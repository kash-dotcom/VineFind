import os
import requests

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"{dest_path} already exists, skipping download.")

# Example usage:
download_file(
    "https://your-cloud-storage.com/path/to/largefile.model",
    "models/largefile.model"
)