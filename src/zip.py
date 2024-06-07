import os
import sys
import py7zr
import multivolumefile
import pathlib
import numpy as np

# DEFAULT_NPZ_PATH = os.path.join(os.getcwd(), 'data', 'processed', 'mh100.npz')

FILE_NAME = 'amex-1M_binary-dataset-[intents]'

DEFAULT_NPZ_PATH = os.path.join(os.getcwd(), 'data', 'processed', f'{FILE_NAME}.npz')


DEFAULT_ZIP_PATH = os.path.join(os.getcwd(), 'data', 'compressed', f'{FILE_NAME}.npz.zip')
# DEFAULT_CSV_PATH = os.path.join(os.getcwd(), 'data', 'zip', 'mh100.csv')
# DEFAULT_OUTPUT_PATH = os.path.join(os.getcwd(), 'data', 'compressed', 'zip-intents-permissions')
DEFAULT_OUTPUT_PATH = os.path.join(os.getcwd(), 'data', 'compressed', 'zip-intents')
# FILENAMES = ['mh100.7z.001', 'mh100.7z.002', 'mh100.7z.003', 'mh100.7z.004', 'mh100.7z.005', 'mh100.7z.006', 'mh100.7z.007','mh100.7z.008']


def zip(path_data):
    # Ensure the extraction directory exists
    os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)
    zip_file_path =os.path.join(DEFAULT_OUTPUT_PATH, f'{os.path.basename(path_data)}.zip')
    try:
        with py7zr.SevenZipFile(zip_file_path , 'w') as archive:
            archive.write(path_data, os.path.basename(path_data))
        print(f"Zipped {path_data} successfully into {zip_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

def zip_multi_volume(file_name='amex-1M_binary-dataset'):
    # Ensure the extraction directory exists
    os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)

    try:
        with open(DEFAULT_ZIP_PATH, 'rb') as file:
            data = file.read()
        
        target = os.path.join(DEFAULT_OUTPUT_PATH, f'{file_name}.7z') #pathlib.Path(DEFAULT_OUTPUT_PATH)
        with multivolumefile.open(target, mode='wb') as vol:
            size = vol.write(data)
            vol.seek(0)
            # with py7zr.SevenZipFile(target_archive, 'w') as archive:
            #     archive.writeall(target, 'target')

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return

def main():
    
    if len(sys.argv) > 2:
        print("Usage: python zip.py [<path_to_zip_file>]")
        sys.exit(1)
    # data_file_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_ZIP_PATH

    # zip(path_data=DEFAULT_NPZ_PATH)
    zip_multi_volume(DEFAULT_ZIP_PATH)

if __name__ == "__main__":
    main()
