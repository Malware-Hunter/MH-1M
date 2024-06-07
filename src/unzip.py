import os
import sys
import py7zr
from tqdm import tqdm
import shutil

DEFAULT_ZIP_NPZ_PATH = os.path.join(os.getcwd(), 'data', 'zip-npz')
DEFAULT_EXTRACT_PATH = os.path.join(os.getcwd(), 'data', 'processed')
FILENAMES = ['mh100.7z.0001', 'mh100.7z.0002', 'mh100.7z.0003', 'mh100.7z.0004', 'mh100.7z.0005', 'mh100.7z.0006', 'mh100.7z.0007','mh100.7z.0008']
# DEFAULT_OUTPUT_PATH = os.path.join(os.getcwd(), 'data', 'test')

def main():

    zip_file = 'result.7z'

    if len(sys.argv) > 2:
        print("Usage: python unzip.py ") # [<path_to_zip_file>]
        sys.exit(1)
    
    # filenames = ['example.7z.001', 'example.7z.002']
    with open(zip_file, 'ab') as outfile:  # append in binary mode
        for fname in FILENAMES:
            full_path = os.path.join(DEFAULT_ZIP_NPZ_PATH, fname)
            with open(full_path, 'rb') as infile:        # open in binary mode also
                outfile.write(infile.read())

    with py7zr.SevenZipFile('result.7z', mode='r') as zip:
        zip.extractall(path=DEFAULT_EXTRACT_PATH)

    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        if os.path.isfile(zip_file):
            os.remove(zip_file)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

if __name__ == "__main__":
    main()
