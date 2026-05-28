# Project: Content-Based Image Retrieval (CBIR) Web Application

This project is a **Content-Based Image Retrieval (CBIR)** web application designed to search and retrieve visually similar images from a database based on a query image uploaded by the user. The application processes images, extracts key visual feature signatures, and ranks matching database images using selected distance metrics.

## Key Features

- **Query Image Upload:** Supports JPG, PNG, and JPEG formats.
- **Advanced Feature Descriptors:**
  - **GLCM (Gray-Level Co-occurrence Matrix):** Extracts texture features such as contrast, dissimilarity, homogeneity, correlation, energy, and ASM.
  - **Haralick Texture Features:** Computes statistical features based on gray-level co-occurrence.
  - **BiT (Binarized Image Features):** Uses biological/taxonomical scale image features.
  - **Fusion:** Concatenates GLCM, Haralick, and BiT features for a comprehensive multi-descriptor signature.
- **Configurable Search Options:**
  - Adjustable slider to select the number of similar images to retrieve (Top K).
  - Multiple distance metrics: **Euclidean**, **Manhattan** (Cityblock), **Chebyshev**, and **Canberra**.
- **Batch Signature Extraction:** Precompute signatures for the entire database using the `extraction.py` script.

## Technologies Used & Main Libraries

- **Python**: Core programming language.
- **Streamlit**: Web application framework for the interactive user interface.
- **NumPy**: Efficient multidimensional array operations and storage of signature files (`.npy`).
- **SciPy**: Efficient computation of distance metrics (`scipy.spatial.distance`).
- **scikit-image (skimage)**: High-quality image processing filters and texture features (`graycomatrix`, `graycoprops`).
- **mahotas**: High-performance computer vision library for Haralick texture feature computation.
- **BiT**: Custom feature extractor for biological taxo-scale signatures.
- **Pillow (PIL)**: Python Imaging Library for image reading, resizing, and color-space conversion.

## Project Structure

Here is the directory structure of the project:

```text
├── app.py                 # Streamlit web application entry point
├── extraction.py          # Script to extract and save features for the dataset
├── requirements.txt       # Python package dependencies
├── data/
│   └── dataset/           # Folder containing the database images
├── descriptors/
│   ├── glcm_rgb.py        # GLCM feature extractor
│   ├── haralick_rgb.py    # Haralick texture feature extractor
│   ├── bit_rgb.py         # BiT feature extractor
│   └── fusion.py          # Feature fusion concatenator
├── distances/
│   └── metrics.py         # Distance metrics implementation (Euclidean, Manhattan, etc.)
├── signatures/
│   ├── GLCM_RGB.npy       # Precomputed GLCM features of the dataset
│   ├── Haralick_RGB.npy   # Precomputed Haralick features of the dataset
│   ├── BiT_RGB.npy        # Precomputed BiT features of the dataset
│   ├── Fusion.npy         # Precomputed Fusion features of the dataset
│   └── paths.npy          # Precomputed relative image file paths
└── utils/
    └── image_loader.py    # Utility to load and preprocess images
```

### Components Description

- **`app.py`**: The frontend Streamlit application providing the web interface. It handles image upload, allows setting execution options, calls feature extraction on the query image, performs similarity searches, and displays similar image results.
- **`extraction.py`**: A batch processing script to extract features from all images located inside `data/dataset/` using all available descriptors and save them as NumPy signature files under `signatures/`.
- **`data/`**: The folder structure to store the database images under the `data/dataset/` directory.
- **`descriptors/`**: Contains Python modules implementing various feature extractors. Each module defines a function `extract_features(image)`.
- **`distances/`**: Computes distances (similarity scores) between feature vectors. Exposes the `get_similar_images` function which compares the query signature to database signatures.
- **`signatures/`**: Stores binary NumPy array files (`.npy`) containing the extracted features of all database images and their relative paths. This allows fast image retrieval without needing to extract features from all database images on every query.
- **`utils/`**: Helper utilities. `image_loader.py` loads PIL images, resizes them to 128x128 pixels, and discards the alpha channel if present.

## Creating and configuring the virtual environment

If you have more than one version of Python installed on your machine, type the command below to force the installation of the venv development environment in version 3.10:

```bash
py -3.10 -m venv venv
```

Activate the virtual environment created:

```bash
.\venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the application

Command to start the application:

```bash
streamlit run app.py
```

# GitHub Repository

You can find the source code for this project on GitHub:
[Streamlit CBIR Project](https://github.com/lucas-ladeira/Projet_CBIR_Streamlit)