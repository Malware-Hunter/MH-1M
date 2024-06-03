import os
import sys
import subprocess
from tqdm import tqdm

DEFAULT_ZIP_PATH = './data/zip/amex-1M_binary-dataset-[intents-permissions-apicalls].zip'
DEFAULT_EXTRACT_PATH = './data/processed/'

def extract_file_with_7z(zip_file_path, extract_to=DEFAULT_EXTRACT_PATH):
    """
    Extract a file using 7z to the specified directory.

    :param zip_file_path: Path to the zip file.
    :param extract_to: Directory to extract files to. Default is './data/processed/'.
    """
    if not os.path.exists(zip_file_path):
        print(f"The file {zip_file_path} does not exist.")
        return
    
    # Ensure the extraction directory exists
    os.makedirs(extract_to, exist_ok=True)

    try:
        process = subprocess.Popen(['7z', 'x', f'-o{extract_to}', zip_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Count total number of lines to approximate the progress bar
        total_lines = sum(1 for line in process.stdout)
        process.stdout.seek(0)

        with tqdm(total=total_lines, desc="Extracting", unit="line") as pbar:
            for line in process.stdout:
                pbar.update(1)
                print(line, end='')
                
        process.wait()  # Wait for the process to complete

        if process.returncode != 0:
            stderr = process.stderr.read()
            raise subprocess.CalledProcessError(process.returncode, process.args, output=None, stderr=stderr)

        print(f"Extracted all files to {extract_to}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to extract {zip_file_path}.\n{e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    if len(sys.argv) > 2:
        print("Usage: python unzip_dataset.py [<path_to_zip_file>]")
        sys.exit(1)
    
    zip_file_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_ZIP_PATH
    extract_file_with_7z(zip_file_path)

if __name__ == "__main__":
    main()