import urllib.request
import os

url = "https://www.gutenberg.org/files/11/11-0.txt"
output_path = "data/alice.txt"

print(f"Downloading {url} to {output_path}...")
try:
    urllib.request.urlretrieve(url, output_path)
    print("Download complete.")
    print(f"File size: {os.path.getsize(output_path)} bytes")
except Exception as e:
    print(f"Error downloading: {e}")
