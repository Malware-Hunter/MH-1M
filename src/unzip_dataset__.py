import zipfile
import os
import sys

DEFAULT_ZIP_PATH = './data/zip/amex-1M_binary-dataset-[intents-permissions-apicalls].zip'
DEFAULT_EXTRACT_PATH = './data/processed/'

def is_valid_zip(file_path):
    """
    Check if the file is a valid zip file.
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            corrupt_file = zip_ref.testzip()
            if corrupt_file is not None:
                print(f"Warning: The zip file is corrupted at {corrupt_file}.")
                return False
            return True
    except zipfile.BadZipFile:
        print(f"Error: The file {file_path} is not a valid zip file or it is corrupted.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while checking the zip file: {e}")
        return False

def unzip_file(zip_file_path, extract_to=DEFAULT_EXTRACT_PATH):
    """
    Unzip a zip file to the specified directory.

    :param zip_file_path: Path to the zip file.
    :param extract_to: Directory to extract files to. Default is './data/processed/'.
    """
    if not os.path.exists(zip_file_path):
        print(f"The file {zip_file_path} does not exist.")
        return

    if not is_valid_zip(zip_file_path):
        print(f"The file {zip_file_path} is not a valid zip file or it is corrupted.")
        return

    # Ensure the extraction directory exists
    os.makedirs(extract_to, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted all files to {extract_to}")
    except zipfile.LargeZipFile:
        print(f"Error: The file {zip_file_path} requires ZIP64 functionality but it is not enabled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    if len(sys.argv) > 2:
        print("Usage: python unzip_dataset.py [<path_to_zip_file>]")
        sys.exit(1)

    zip_file_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_ZIP_PATH
    unzip_file(zip_file_path)

if __name__ == "__main__":
    main()
