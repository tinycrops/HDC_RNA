import os
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from .data_loader import RNADataLoader
from .rna_hdc_model import RNAHDC3DPredictor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RNA HDC 3D folding model')
    
    parser.add_argument('--data_dir', type=str, default='../stanford-rna-3d-folding',
                        help='Directory containing RNA data files')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of sequences to use for training')
    parser.add_argument('--sample', action='store_true',
                        help='Use a small sample of data for quick testing')
    parser.add_argument('--hdc_dimensions', type=int, default=10000,
                        help='Dimensionality of hypervectors')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {args.device}")
    print(f"HDC dimensions: {args.hdc_dimensions}")
    
    # Load data
    data_loader = RNADataLoader(args.data_dir)
    
    if args.sample:
        print("Using sample data for quick testing")
        sequences_df, coordinates_df = data_loader.get_sample_data(max_sequences=5)
    else:
        print(f"Preparing training data{' (limited to ' + str(args.max_sequences) + ' sequences)' if args.max_sequences else ''}")
        sequences_df, coordinates_df = data_loader.prepare_training_data(max_sequences=args.max_sequences)
    
    print(f"Loaded {len(sequences_df)} sequences and {len(coordinates_df)} coordinate records")
    
    # Initialize model
    model = RNAHDC3DPredictor(hdc_dimensions=args.hdc_dimensions, device=args.device)
    
    # Prepare training data
    print("Preparing training data loader")
    train_loader = model.prepare_data(sequences_df, coordinates_df, batch_size=args.batch_size)
    
    # Train model
    print(f"Training model for {args.epochs} epochs")
    losses = model.train(train_loader, epochs=args.epochs)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'rna_hdc_model.pt')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
    plt.close()

if __name__ == '__main__':
    main() 