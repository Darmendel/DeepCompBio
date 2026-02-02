# Predicting RNAcompete Binding from RNA Bind-n-Seq Data

A deep learning project for predicting RNAcompete binding outcomes based on RNA Bind-n-Seq (RBNS) data using a Convolutional Neural Network (CNN) architecture.

## Overview

This project implements a CNN-based model to predict RNA-protein binding interactions. The model is trained on RBNS data from different RNA concentrations and makes predictions on RNAcompete sequences.

## Model Architecture

The `CNNRBAModel` consists of:
- **Convolutional Layer**: 1D convolution with 700 filters, kernel size 5
- **Max Pooling**: Reduces sequence dimension by factor of 40
- **Hidden Layer**: Fully connected layer with 700 units
- **Output Layer**: Single sigmoid output for binary classification

## Requirements

- Python version 3.7-3.9
- Required packages are listed in the `requirements.txt` file

Install dependencies using:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 1.9.0
- NumPy 1.21.6
- scikit-learn (via joblib)

## How to Run

To run the project, follow these steps:

1. Open your terminal
2. Navigate to the project directory
3. Run the following command:

```bash
python3 main.py <ofile> <RNCMPT> <input> <RBNS1> <RBNS2> ... <RBNSk>
```

**Arguments:**
- `<ofile>`: Path to the output file where predictions will be saved
- `<RNCMPT>`: Path to the `RNAcompete_sequence.txt` file
- `<input>`: Path to the `RBPi_input.seq` file
- `<RBNS1>`, `<RBNS2>`, ..., `<RBNSk>`: Paths to RBNS files corresponding to different concentrations (j1nM, j2nM, ..., jknM)

**Example:**

```bash
python3 main.py results/output.txt data/RNAcompete_sequence.txt data/RBPi_input.seq data/RBPi_j1nM.seq data/RBPi_j2nM.seq data/RBPi_j5nM.seq
```

This command will train the model on the provided RBNS data and generate binding predictions for the RNAcompete sequences.

**Note:** Ensure you have the necessary files in the specified locations before running the command.

## Programmatic Usage

You can also use the modules programmatically:

```python
from train_model import train_model
from data_loader import DataGenerator
from main import predict_rna_compete

# Load RNAcompete sequences
rna_seqs = DataGenerator.rna_compete_to_tensor('data/RNAcompete_sequence.txt')

# Train model on RBNS files (input file first, then concentration files)
rbns_files = ['data/RBPi_input.seq', 'data/RBPi_j1nM.seq', 'data/RBPi_j5nM.seq']
model = train_model(rbns_files, max_epochs=10, show_performance=True)

# Make predictions
predictions = predict_rna_compete(model, rna_seqs)

# Save predictions
import numpy as np
np.savetxt('predictions.txt', predictions.numpy())
```

## Data Format

### RBNS Files
RBNS files should contain RNA sequences with each sequence split across two consecutive lines:
```
AUCGAUCGAUCG
CGAUCGAUCGAU
GCUAGCUAGCUA
AUCGUCGAUCGA
```

The code reads pairs of lines and concatenates them to form complete sequences.

### RNAcompete Files
RNAcompete files should contain one RNA sequence per line:
```
AUCGAUCGAUCG
CGAUCGAUCGAU
GCUAGCUAGCUA
```

## Training Details

The training process:
1. Loads RBNS data (up to 1M samples) - uses first file for negative samples, last file for positive samples
2. Automatically pads/truncates sequences to 40 nucleotides
3. Splits data into train/validation sets (70/30)
4. Trains using Adam optimizer (learning rate: 0.001) with Binary Cross-Entropy loss
5. Supports early stopping based on `max_runtime` parameter
6. Evaluates on validation set each epoch (if `show_performance=True`)

## Training Parameters

The `train_model` function accepts:
- `rbns_files`: List of RBNS file paths
- `max_epochs`: Maximum number of training epochs (default: 2)
- `max_runtime`: Maximum training time in seconds (default: 3600)
- `show_performance`: Whether to print training metrics (default: False)

## Features

- **Automatic Padding**: Sequences are automatically padded/truncated to 40 nucleotides
- **One-Hot Encoding**: RNA sequences encoded as 4-dimensional vectors (A, C, G, T/U)
- **Train/Validation Split**: 70/30 split for model evaluation
- **Batch Processing**: Efficient batch prediction with configurable batch size (default: 64)
- **Time Constraints**: Supports maximum training time limits
- **RNA/DNA Compatibility**: Automatically converts 'U' to 'T' for compatibility

## Output

Predictions are saved as a text file with one prediction per line, representing the binding probability (0-1) for each RNAcompete sequence. Higher values indicate stronger predicted binding affinity.

## Project Structure

```
.
├── main.py              # Main entry point and CLI
├── train_model.py       # Training and evaluation functions
├── model.py             # CNN model architecture
├── data_loader.py       # Data loading and preprocessing
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Important Notes

- Sequences containing 'U' (RNA) are automatically converted to 'T' (DNA) for processing
- Unknown nucleotides are encoded as 'N' with equal probabilities [0.25, 0.25, 0.25, 0.25]
- The model uses GPU if available (PyTorch default behavior)
- Training will stop early if `max_runtime` is exceeded
- The first RBNS file is used for negative samples, the last file for positive samples
- All sequences are automatically padded or truncated to exactly 40 nucleotides
