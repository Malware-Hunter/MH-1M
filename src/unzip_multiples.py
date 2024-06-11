
import os
import sys
import py7zr

def concatenate_parts(parts_directory, output_file):
    """
    Concatenates parts into a single file.

    Parameters:
    parts_directory (str): Directory containing the parts.
    output_file (str): Path to the output file.
    """
    with open(output_file, 'wb') as outfile:
        for part in sorted(os.listdir(parts_directory)):
            print('Part: ', part)
            part_path = os.path.join(parts_directory, part)
            if os.path.isfile(part_path):
                with open(part_path, 'rb') as infile:
                    outfile.write(infile.read())

def unzip_concatenated_file(concatenated_file, extract_to):
    """
    Unzips the concatenated .7z file.

    Parameters:
    concatenated_file (str): Path to the concatenated .7z file.
    extract_to (str): Directory to extract the contents to.
    """
    try:
        with py7zr.SevenZipFile(concatenated_file, mode='r') as archive:
            archive.extractall(path=extract_to)
        print(f"Extracted {concatenated_file} to {extract_to}")
    except py7zr.Bad7zFile:
        print(f"The file {concatenated_file} is not a valid 7z file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python unzip_parts.py <parts_directory> <concatenated_output_file> <extract_to_directory>")
        sys.exit(1)
    
    parts_directory = sys.argv[1]
    concatenated_output_file = sys.argv[2]
    extract_to_directory = sys.argv[3]
    
    # Ensure the extract to directory exists
    os.makedirs(extract_to_directory, exist_ok=True)
    
    print('Concatenate parts...')
    concatenate_parts(parts_directory, concatenated_output_file)
    print('Unzip file...')
    unzip_concatenated_file(concatenated_output_file, extract_to_directory)

if __name__ == "__main__":
    main()

    # python unzip_parts.py /path/to/parts /path/to/concatenated_output.zip /path/to/extract_to

