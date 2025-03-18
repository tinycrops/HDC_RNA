import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from .data_loader import RNADataLoader
from .rna_hdc_model import RNAHDC3DPredictor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict RNA 3D structure using HDC model')
    
    parser.add_argument('--data_dir', type=str, default='../stanford-rna-3d-folding',
                        help='Directory containing RNA data files')
    parser.add_argument('--model_path', type=str, default='./models/rna_hdc_model.pt',
                        help='Path to trained model file')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    parser.add_argument('--target_id', type=str, default=None,
                        help='Target ID for prediction (if None, use validation sequences)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='RNA sequence for prediction (if provided, overrides target_id)')
    parser.add_argument('--hdc_dimensions', type=int, default=10000,
                        help='Dimensionality of hypervectors (must match trained model)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for prediction (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate 3D visualization of predicted structure')
    
    return parser.parse_args()

def plot_rna_structure(coords, sequence, title, output_path):
    """
    Generate a 3D plot of RNA structure.
    
    Args:
        coords (np.ndarray): 3D coordinates for each nucleotide
        sequence (str): RNA sequence
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
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each nucleotide
    for i, (x, y, z) in enumerate(coords):
        nucleotide = sequence[i]
        color = colors.get(nucleotide, 'black')
        
        # Plot nucleotide as a point
        ax.scatter(x, y, z, color=color, s=50)
        
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

def save_predictions(coords, sequence, output_path):
    """
    Save predicted coordinates to CSV file.
    
    Args:
        coords (np.ndarray): 3D coordinates for each nucleotide
        sequence (str): RNA sequence
        output_path (str): Path to save the CSV
    """
    # Create DataFrame for predictions
    data = []
    
    for i, (x, y, z) in enumerate(coords):
        data.append({
            'position': i + 1,
            'nucleotide': sequence[i],
            'x': x,
            'y': y,
            'z': z
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")

def main():
    """Main prediction function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {args.device}")
    
    # Initialize model
    model = RNAHDC3DPredictor(hdc_dimensions=args.hdc_dimensions, device=args.device)
    
    # Load trained model
    print(f"Loading model from {args.model_path}")
    model.load_model(args.model_path)
    
    # Get sequence to predict
    if args.sequence:
        # Use provided sequence
        sequence = args.sequence
        target_id = "custom_sequence"
    elif args.target_id:
        # Get sequence from data using target_id
        data_loader = RNADataLoader(args.data_dir)
        
        # Try validation sequences first
        val_sequences, _ = data_loader.load_validation_data()
        target_seq = val_sequences[val_sequences['target_id'] == args.target_id]
        
        if len(target_seq) == 0:
            # Try test sequences
            test_sequences = data_loader.load_test_data()
            target_seq = test_sequences[test_sequences['target_id'] == args.target_id]
            
        if len(target_seq) == 0:
            # Try train sequences
            train_sequences, _ = data_loader.load_train_data()
            target_seq = train_sequences[train_sequences['target_id'] == args.target_id]
            
        if len(target_seq) == 0:
            raise ValueError(f"Target ID '{args.target_id}' not found in data")
            
        sequence = target_seq['sequence'].iloc[0]
        target_id = args.target_id
    else:
        # Use first validation sequence
        data_loader = RNADataLoader(args.data_dir)
        val_sequences, _ = data_loader.load_validation_data()
        target_id = val_sequences['target_id'].iloc[0]
        sequence = val_sequences['sequence'].iloc[0]
    
    # Make prediction
    print(f"Predicting 3D structure for sequence (length {len(sequence)})")
    coords = model.predict_coordinates(sequence)
    
    # Save predictions
    output_csv = os.path.join(args.output_dir, f"{target_id}_prediction.csv")
    save_predictions(coords, sequence, output_csv)
    
    # Optionally visualize the predicted structure
    if args.visualize:
        output_plot = os.path.join(args.output_dir, f"{target_id}_structure.png")
        plot_rna_structure(coords, sequence, f"Predicted structure for {target_id}", output_plot)

if __name__ == '__main__':
    main() 