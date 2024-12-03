# Integrating Genetic Information for Alzheimer's Diagnosis through MRI Interpretation

This repository contains the code for the paper "Integrating Genetic Information for Alzheimer's Diagnosis through MRI Interpretation," presented at the IEEE International Conference on Biomedical and Health Informatics (BHI).[PAPER website](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10313442).

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)

## Introduction

This project aims to enhance the early diagnosis of Alzheimer's disease by integrating genetic information with MRI data. By leveraging multimodal AI techniques, we provide a comprehensive approach to predict and interpret Alzheimer's disease progression.

## Usage

To run the code, install the required modules: `torch`, `nibabel`, `pandas`, `scipy`.

To use the code, follow these steps:

1. Prepare your dataset as specified in the [Dataset](#dataset) section.
2. Train and evaluate the model:
    ```bash
    python train.py --epoch [EPOCH_NUMBER] --save_path [SAVE_PATH]
    ```

## Dataset

The dataset used in this project combines MRI scans and genetic information.

You can download the MRI and genetic datasets from the [ADNI website](https://adni.loni.usc.edu/).

For MRI, bias correction, linear registration using the ICMB MNI 152, and skull stripping were performed using Freesurfer.

## Results

The results of our experiments demonstrate the effectiveness of integrating genetic and MRI data for early diagnosis of Alzheimer's disease.  
Detailed results and interpretation are available in the paper.
