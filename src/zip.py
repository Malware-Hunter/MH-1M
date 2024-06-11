import os
import py7zr
import sys

def zip_files(npz_path, output_path):
    # Ensure the extraction directory exists
    os.makedirs(output_path, exist_ok=True)
    zip_file_path = os.path.join(output_path, f'{os.path.basename(npz_path)}.7z')
    try:
        with py7zr.SevenZipFile(zip_file_path, 'w') as archive:
            archive.writeall(npz_path, os.path.basename(npz_path))
        print(f"Zipped {npz_path} successfully into {zip_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python zip.py <path_to_zip_file> <output_path>")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    output_path = sys.argv[2]
    
    zip_files(npz_path, output_path)

if __name__ == "__main__":
    main()

    # python 