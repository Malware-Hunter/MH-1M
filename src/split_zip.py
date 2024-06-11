import os
import sys
import py7zr
import multivolumefile
import pathlib

def zip_multi_volume(zip_path, output_path, file_name='amex-1M_binary-dataset', volume_size=100 * 1024 * 1024): # volume_size=100 * 1024 * 1024
    # Ensure the extraction directory exists
    os.makedirs(output_path, exist_ok=True)

    try:
        target = os.path.join(output_path, f'{file_name}.7z')

        # Open the source file and create a multivolume file
        with open(zip_path, 'rb') as file:
            print('ZIP file readed')
            with multivolumefile.open(target, mode='wb') as vol:
                while True:
                    data = file.read(100)
                    if not data:
                        break
                    vol.write(data)

        # Create the 7z archive using the multivolume file
        with py7zr.SevenZipFile(target, 'w') as archive:
            archive.writeall(zip_path, os.path.basename(zip_path))

        print(f"Zipped {zip_path} successfully into {target} with volumes of {volume_size // (1024 * 1024)}MB")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def split_zip_file(zip_path, output_path, part_size=100 * 1024 * 1024):
    """
    Splits the given zip file into smaller parts of fixed size.

    Parameters:
    zip_path (str): Path to the input zip file.
    output_path (str): Path to the directory where parts will be stored.
    part_size (int): Size of each part in bytes. Default is 100MB.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    try:
        part_num = 100
        with open(zip_path, 'rb') as file:
            while True:
                # Read a chunk of the specified part size
                data = file.read(part_size)
                if not data:
                    break
                # Define the target path for the current part
                part_file_path = os.path.join(output_path, f'{os.path.basename(zip_path)}.part{part_num}')
                with open(part_file_path, 'wb') as part_file:
                    part_file.write(data)
                part_num += 1

        print(f"Split {zip_path} successfully into parts stored in {output_path}, each with a size of {part_size // (1024 * 1024)}MB")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python zip.py <path_to_zip_file> <output_path>")
        sys.exit(1)
    
    zip_path = sys.argv[1]
    output_path = sys.argv[2]
    
    file_name=zip_path.replace('.7z', '').replace('.npz', '')
    print(f'file name: {file_name}')
    # zip_multi_volume(zip_path, output_path, file_name=file_name)
    split_zip_file(zip_path, output_path)


if __name__ == "__main__":
    main()

    # python src/split_zip.py data/compressed/amex-1M-\[intents-permissions-opcodes-apicalls\].npz.7z data/compressed/zip-intents-permissions-apicalls/
