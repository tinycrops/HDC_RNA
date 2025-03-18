#!/usr/bin/env python
"""
Quick demonstration of HDC-based RNA 3D structure prediction.
This script uses a small sample of data to train a model and visualize predictions.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .data_loader import RNADataLoader
from .rna_hdc_model import RNAHDC3DPredictor

def create_directories():
    """Create necessary directories."""
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./predictions', exist_ok=True)

def plot_rna_3d(sequence, coords, title, output_path):
    """
    Plot RNA 3D structure.
    
    Args:
        sequence (str): RNA sequence
        coords (np.ndarray): 3D coordinates for each nucleotide
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    # Map nucleotides to colors
    colors = {
        'A': 'red',
        'C': 'blue',
        'G': 'green',
        'U': 'orange'
    }
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each nucleotide
    for i, (x, y, z) in enumerate(coords):
        nucleotide = sequence[i]
        color = colors.get(nucleotide, 'black')
        
        # Plot nucleotide as a point
        ax.scatter(x, y, z, color=color, s=100)
        
        # Add nucleotide label
        ax.text(x, y, z, f"{i+1}:{nucleotide}", size=8)
        
        # Connect to next nucleotide with a line
        if i < len(coords) - 1:
            next_x, next_y, next_z = coords[i+1]
            ax.plot([x, next_x], [y, next_y], [z, next_z], 'k-', alpha=0.5)
    
    # Add legend
    for nucleotide, color in colors.items():
        ax.scatter([], [], [], color=color, label=nucleotide)
    ax.legend()
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def compare_actual_vs_predicted(sequence, actual_coords, predicted_coords, title, output_path):
    """
    Plot actual vs predicted RNA 3D structure.
    
    Args:
        sequence (str): RNA sequence
        actual_coords (np.ndarray): Actual 3D coordinates
        predicted_coords (np.ndarray): Predicted 3D coordinates
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    # Map nucleotides to colors
    colors = {
        'A': 'red',
        'C': 'blue',
        'G': 'green',
        'U': 'orange'
    }
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each nucleotide
    for i, ((x1, y1, z1), (x2, y2, z2)) in enumerate(zip(actual_coords, predicted_coords)):
        nucleotide = sequence[i]
        color = colors.get(nucleotide, 'black')
        
        # Plot actual position
        ax.scatter(x1, y1, z1, color=color, s=100, alpha=0.8, marker='o')
        
        # Plot predicted position
        ax.scatter(x2, y2, z2, color=color, s=100, alpha=0.4, marker='^')
        
        # Draw a line between actual and predicted
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'k--', alpha=0.3)
        
        # Add nucleotide label
        ax.text(x1, y1, z1, f"{i+1}:{nucleotide}", size=8)
    
    # Add legend
    for nucleotide, color in colors.items():
        ax.scatter([], [], [], color=color, label=nucleotide)
    ax.scatter([], [], [], color='k', alpha=0.8, marker='o', label='Actual')
    ax.scatter([], [], [], color='k', alpha=0.4, marker='^', label='Predicted')
    ax.legend()
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    
    print(f"Comparison visualization saved to {output_path}")

def main():
    """Run a sample demonstration."""
    create_directories()
    
    print("Running HDC RNA 3D structure prediction demonstration")
    
    # Use CPU for the demo
    device = 'cpu'
    
    # Load a small sample of data
    print("Loading sample data...")
    data_loader = RNADataLoader('./stanford-rna-3d-folding')
    sequences_df, coordinates_df = data_loader.get_sample_data(max_sequences=2)
    
    # Initialize model with smaller dimensions for faster demo
    print("Initializing model...")
    model = RNAHDC3DPredictor(hdc_dimensions=1000, device=device)
    
    # Prepare data for training
    print("Preparing data...")
    train_loader = model.prepare_data(sequences_df, coordinates_df, batch_size=16)
    
    # Train model for just a few epochs
    print("Training model (quick demo)...")
    losses = model.train(train_loader, epochs=3)
    
    # Save model
    model.save_model('./models/demo_model.pt')
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('./predictions/training_loss.png')
    plt.close()
    
    # Select a sequence for prediction
    seq_id = sequences_df['target_id'].iloc[0]
    sequence = sequences_df['sequence'].iloc[0]
    
    print(f"Predicting structure for sequence: {seq_id}")
    
    # Predict 3D coordinates
    predicted_coords = model.predict_coordinates(sequence)
    
    # Get actual coordinates for comparison
    seq_coords = coordinates_df[coordinates_df['ID'].str.startswith(seq_id)]
    seq_coords = seq_coords.sort_values('resid')
    
    # Extract actual coordinates
    actual_coords = []
    for _, row in seq_coords.iterrows():
        if len(actual_coords) >= len(sequence):
            break
        actual_coords.append([row['x_1'], row['y_1'], row['z_1']])
    
    actual_coords = np.array(actual_coords)
    
    # Ensure same length for comparison
    min_len = min(len(actual_coords), len(predicted_coords))
    actual_coords = actual_coords[:min_len]
    predicted_coords = predicted_coords[:min_len]
    
    # Plot predicted structure
    plot_rna_3d(
        sequence[:min_len], 
        predicted_coords, 
        f"Predicted structure for {seq_id}",
        './predictions/predicted_structure.png'
    )
    
    # Plot actual structure
    plot_rna_3d(
        sequence[:min_len], 
        actual_coords, 
        f"Actual structure for {seq_id}",
        './predictions/actual_structure.png'
    )
    
    # Compare actual vs predicted
    compare_actual_vs_predicted(
        sequence[:min_len],
        actual_coords,
        predicted_coords,
        f"Actual vs Predicted structure for {seq_id}",
        './predictions/comparison.png'
    )
    
    print("\nDemonstration completed!")
    print("Check the 'predictions' directory for visualizations.")

if __name__ == "__main__":
    main() 