# ADC
The main architecture and loss function for the adversarial single-domain generalization method has been uploaded.
# Data link
1. Paderborn University (PU) dataset: [https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/](https://mb.uni-paderborn.de/en/kat/research/kat-datacenter/bearing-datacenter/data-sets-and-download)
2. Huazhong University of Science and Technology (HUST) dataset: https://github.com/CHAOZHAO-1/HUSTbearing-dataset.
3. Beijing University of Technology (BJUT) dataset: https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset/tree/master
4. Shandong University of Science and Technology(SDUST) dataset: https://github.com/JRWang-SDUST/SDUST-Dataset.git
5. Real Factory Bearing (RFB) dataset: https://drive.google.com/drive/folders/1DHthZwWF6UCn5ukUCBAqJUe1uBwbxXX8?usp=share_link
# Data preprocessing
The raw input data consists of **vibration acceleration signals**. We do not apply any additional preprocessing steps such as denoising or normalization. 

- Each sample is extracted by segmenting the original signal into fixed-length windows.
- A **Fast Fourier Transform (FFT)** is applied to each segment during data loading, implemented in `load_data.py`.

This ensures that the entire pipeline is simple, reproducible, and focused on frequency-domain representations only.

👉 Download the segmented time-domain data here:  
[📥 Click to download time-domain dataset](https://your-download-link.com)
# Environment Setup
This project is developed and tested under Python 3.9 with PyTorch ≥ 1.12.

We recommend using `conda` to manage the environment for consistent dependencies.

# Project Structure
ADC

├── module/ # Loss functions and modules

├── load_data.py # Data loading and FFT

├── construct_loader.py # Dataloader builder

├── main.py # Training and evaluation

└── README.md
#  Quick Start

We provide a sample dataset and a ready-to-run script so that users can quickly reproduce the results.

1. **Preparing data**
- Option 1: Use the dataset already included in the `./data/` folder (if you downloaded the full repository ZIP).
- Option 2: Download the data manually from: [📥 External download link](https://your-download-link.com)

2. **Place the dataset in the `./data/` directory** (create this folder if it doesn't exist).

3. **Clone this repository**
   
4. **Run the main file to start training and evaluation**
   
