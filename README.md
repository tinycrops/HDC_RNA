# HDC-RNA: Hyperdimensional Computing for RNA 3D Structure Prediction

This project applies Hyperdimensional Computing (HDC) techniques to predict the 3D structure of RNA molecules. By encoding RNA sequences in high-dimensional space and leveraging HDC operations, we aim to improve the accuracy and efficiency of RNA structure prediction.

## Overview

Hyperdimensional Computing is a brain-inspired computing paradigm that represents information with high-dimensional vectors (typically 10,000+ dimensions). The key HDC operations include:

- **Binding**: Create associations between concepts
- **Bundling**: Combine multiple elements
- **Permutation**: Encode sequence information

These operations enable holistic representation of complex data while maintaining robustness to noise and errors, making HDC particularly well-suited for biological sequence analysis.

## Features

- RNA sequence encoding using hyperdimensional vectors
- Position-specific nucleotide representation
- N-gram context encoding for capturing local patterns
- Neural network-based coordinate prediction
- 3D structure visualization
- Easy-to-use command line interface

## Installation

Clone the repository:

```bash
git clone https://github.com/tinycrops/hdc-rna.git
cd hdc-rna
```

Install dependencies:

```bash
pip install -r hdc_rna/requirements.txt
```

## Data

This project uses the Stanford RNA 3D Folding dataset, which includes:

- RNA sequences with their 3D coordinates
- Training, validation, and test sets
- Supporting multiple sequence alignments (MSAs)

The dataset can be found in the `stanford-rna-3d-folding` directory.

## Usage

The project provides a simple command-line interface for all operations:

### Quick Demonstration

Run a quick demo with sample data:

```bash
python run_hdc_rna.py demo
```

### Testing HDC Implementation

Test the basic HDC operations:

```bash
python run_hdc_rna.py test
```

### Training

Train a model on the full dataset:

```bash
python run_hdc_rna.py train --data_dir stanford-rna-3d-folding --output_dir models
```

For faster training with limited data:

```bash
python run_hdc_rna.py train --data_dir stanford-rna-3d-folding --max_sequences 10
```

### Prediction

Predict the 3D structure of an RNA sequence:

```bash
python run_hdc_rna.py predict --model_path models/rna_hdc_model.pt --sequence GGGUGCUCAGUACGAGAGGAACCGCACCC --visualize
```

Predict a structure from the dataset by ID:

```bash
python run_hdc_rna.py predict --model_path models/rna_hdc_model.pt --target_id R1107 --visualize
```

## How It Works

1. **Encoding**: RNA sequences are encoded as hypervectors:
   - Each nucleotide (A, C, G, U) has its own random hypervector
   - Position information is incorporated using position-specific vectors
   - Local context is captured with n-gram encoding

2. **Neural Mapping**: A neural network maps from the HDC space to 3D coordinates:
   - Input: High-dimensional representation of RNA sequence
   - Output: x, y, z coordinates for each nucleotide

3. **Visualization**: 3D structures are rendered for analysis:
   - Color-coded by nucleotide type
   - Connections shown between adjacent nucleotides
   - Comparison between predicted and actual structures (when available)

## Benefits of HDC for RNA Structure Prediction

- **Robustness**: Maintains accuracy despite mutations or sequencing errors
- **Efficiency**: Requires fewer training examples than traditional deep learning
- **Holistic Representation**: Captures relationships between distant parts of the sequence
- **Interpretability**: More transparent approach than black-box deep learning models
- **Parallelizability**: HDC operations can be highly parallelized

## Future Improvements

- Incorporate evolutionary information from MSAs
- Add physical constraints from RNA folding energetics
- Implement end-to-end differentiable HDC operations
- Integrate with existing RNA structure prediction tools
- Hardware acceleration for HDC operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stanford RNA 3D Folding competition for providing the dataset
- The hyperdimensional computing research community for foundational work
- Contributors to open-source libraries used in this project (PyTorch, NumPy, etc.)