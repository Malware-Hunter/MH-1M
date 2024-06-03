# Usage Instructions

## Table of Contents

- [Prerequisites](#prerequisites)
- [Unzipping the Dataset](#unzipping-the-dataset)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)


## Prerequisites

### Installing 7-Zip

To use the `unzip_dataset.py` script, you need to have 7-Zip installed on your system. Follow the instructions below to install it on your operating system.

#### Windows

1. Download and install 7-Zip from [7-Zip's official website](https://www.7-zip.org/).
2. During installation, ensure that the `7z.exe` executable is added to your system's PATH.

#### Linux

1. Install the `p7zip-full` package:
   ```sh
   sudo apt-get install p7zip-full


### Setting Up Python 3.12 Environment
Ensure you have Python 3.12 installed. You can download and install it from the official Python website.

### Install VS Code:

Download and install Visual Studio Code from VS Code's official website.
Install Python Extension for VS Code:

### Installing Jupyter Lab on VS Code
To set up Jupyter Lab in Visual Studio Code (VS Code), follow these steps:


## Unzipping the Dataset

Before you begin with data preparation, you need to unzip the dataset. Follow the steps below to unzip the dataset using the provided `unzip_dataset.py` script.

1. **Ensure the zip file is available** in the `./zip/` directory or any other location of your choice.

2. **Run the script** to unzip the dataset:

   ```sh
   python unzip_dataset.py <path_to_zip_file>