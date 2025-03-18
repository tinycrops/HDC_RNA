# Hyperdimensional Computing for RNA 3D Structure Prediction

This project applies Hyperdimensional Computing (HDC) techniques to the problem of RNA 3D structure prediction. By encoding RNA sequences in high-dimensional vectors and leveraging HDC operations, we aim to predict the 3D coordinates of RNA nucleotides more efficiently and accurately than traditional methods.

## Overview

Hyperdimensional Computing is a brain-inspired computing paradigm that operates on high-dimensional vectors (thousands of dimensions). The key operations include:

- **Binding**: Element-wise multiplication, used to associate features with values
- **Bundling**: Vector addition, used to combine multiple concepts
- **Permutation**: Cyclic shifting, used to represent sequence order

These operations allow us to encode RNA sequences with their structural context in a way that preserves the complex relationships between nucleotides while providing noise robustness.

## Approach

Our approach combines HDC with neural networks:

1. **RNA Encoding**: Represent RNA sequences as high-dimensional vectors
   - Each nucleotide (A, C, G, U) has its own random hypervector
   - Position information is encoded using position-specific hypervectors
   - N-gram context captures local structural patterns

2. **Coordinate Prediction**: Map from HDC space to 3D coordinates
   - Neural network learns the mapping from hypervectors to 3D space
   - Captures the complex relationship between sequence and structure

3. **Structural Knowledge**: Optionally predict secondary structure elements
   - Helps guide the 3D coordinate prediction
   - Adds biological constraints to the prediction

## Directory Structure

- `hdc_utils.py`: Core HDC operations and encoding utilities
- `rna_hdc_model.py`: The HDC-based RNA structure prediction model
- `data_loader.py`: Utilities for loading and preprocessing RNA data
- `train_model.py`: Script for training the model
- `predict.py`: Script for predicting 3D structures with the trained model

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train the model on the Stanford RNA 3D folding dataset:

```bash
python train_model.py --data_dir ../stanford-rna-3d-folding --output_dir ./models
```

For quick testing with a small sample:

```bash
python train_model.py --sample --epochs 2
```

### Prediction

Predict the 3D structure of an RNA sequence:

```bash
python predict.py --model_path ./models/rna_hdc_model.pt --visualize
```

Predict a specific sequence by ID:

```bash
python predict.py --target_id R1107 --visualize
```

Or provide a custom sequence:

```bash
python predict.py --sequence GGGUGCUCAGUACGAGAGGAACCGCACCC --visualize
```

## Advantages of HDC for RNA Structure Prediction

1. **Robustness to Noise**: Maintains accuracy despite small mutations or errors
2. **Parallelizability**: HDC operations are highly parallelizable
3. **Efficiency**: Requires fewer training examples than traditional deep learning
4. **Interpretability**: Provides a more transparent approach than black-box models
5. **Holistic Representation**: Captures complex dependencies between remote parts of the sequence

## Future Improvements

- Incorporate evolutionary information from multiple sequence alignments (MSAs)
- Add physical constraints based on RNA folding energetics
- Integrate tertiary structure motif recognition
- Implement end-to-end differentiable HDC operations for more efficient training 