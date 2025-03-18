import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from hdc_utils import HDC

class RNAHDC3DPredictor:
    """
    RNA 3D structure predictor using Hyperdimensional Computing.
    
    This model combines HDC encoding of RNA sequences with neural networks
    to predict 3D coordinates of RNA nucleotides.
    """
    
    def __init__(self, hdc_dimensions=10000, device='cpu'):
        """
        Initialize the RNA HDC 3D predictor.
        
        Args:
            hdc_dimensions (int): Dimensionality of hypervectors
            device (str): Device to use for computations ('cpu' or 'cuda')
        """
        self.device = device
        self.hdc = HDC(dimensions=hdc_dimensions, device=device)
        
        # Neural network to map from HDC space to 3D coordinates
        self.coordinate_predictor = nn.Sequential(
            nn.Linear(hdc_dimensions, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # x, y, z coordinates
        ).to(device)
        
        # Secondary structure awareness module (optional)
        self.ss_predictor = nn.Sequential(
            nn.Linear(hdc_dimensions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Stem, loop, bulge, or other
        ).to(device)
        
        self.optimizer = optim.Adam(
            list(self.coordinate_predictor.parameters()) + 
            list(self.ss_predictor.parameters()), 
            lr=0.001
        )
        
        self.coordinate_loss = nn.MSELoss()
        self.ss_loss = nn.CrossEntropyLoss()
    
    def encode_rna_sequence(self, sequence, use_ngrams=True):
        """
        Encode an RNA sequence using HDC.
        
        Args:
            sequence (str): RNA sequence
            use_ngrams (bool): Whether to use n-gram encoding
            
        Returns:
            torch.Tensor: Encoded RNA sequence
        """
        if use_ngrams:
            return self.hdc.encode_sequence_with_ngrams(sequence, n=3)
        else:
            return self.hdc.encode_sequence(sequence)
    
    def prepare_data(self, sequences_df, coordinates_df, batch_size=32):
        """
        Prepare data for training or prediction.
        
        Args:
            sequences_df (pd.DataFrame): DataFrame with RNA sequences
            coordinates_df (pd.DataFrame): DataFrame with 3D coordinates
            batch_size (int): Batch size for training
            
        Returns:
            DataLoader: Data loader for training
        """
        # First, match sequences with their coordinates
        data = []
        
        for _, seq_row in tqdm(sequences_df.iterrows(), desc="Encoding sequences"):
            seq_id = seq_row['target_id']
            sequence = seq_row['sequence']
            
            # Filter coordinates for this sequence
            seq_coords = coordinates_df[coordinates_df['ID'].str.startswith(seq_id)]
            
            if len(seq_coords) == 0:
                continue
                
            # Encode the sequence
            encoded_seq = self.encode_rna_sequence(sequence)
            
            # Get coordinates for each nucleotide
            for _, coord_row in seq_coords.iterrows():
                position = int(coord_row['resid']) - 1  # Convert to 0-indexed
                
                # Get coordinates for this nucleotide
                coords = torch.tensor([
                    coord_row['x_1'], 
                    coord_row['y_1'], 
                    coord_row['z_1']
                ], dtype=torch.float32)
                
                # Encode position-specific nucleotide
                nucleotide = sequence[position] if position < len(sequence) else None
                
                if nucleotide is None:
                    continue
                    
                position_vector = self.hdc.encode_nucleotide(nucleotide, position)
                
                # Combine sequence context with position-specific information
                final_vector = self.hdc.bind(encoded_seq, position_vector)
                
                data.append((final_vector, coords))
        
        # Convert to tensors
        X = torch.stack([item[0] for item in data])
        y = torch.stack([item[1] for item in data])
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, train_loader, epochs=10):
        """
        Train the HDC RNA 3D predictor.
        
        Args:
            train_loader (DataLoader): Training data loader
            epochs (int): Number of training epochs
            
        Returns:
            list: Training losses
        """
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                predicted_coords = self.coordinate_predictor(batch_x)
                
                # Compute loss
                loss = self.coordinate_loss(predicted_coords, batch_y)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
        return losses
    
    def predict_coordinates(self, sequence):
        """
        Predict 3D coordinates for an RNA sequence.
        
        Args:
            sequence (str): RNA sequence
            
        Returns:
            np.ndarray: Predicted 3D coordinates for each nucleotide
        """
        self.coordinate_predictor.eval()
        coordinates = []
        
        # Encode the entire sequence
        encoded_seq = self.encode_rna_sequence(sequence)
        
        with torch.no_grad():
            for i, nucleotide in enumerate(sequence):
                # Encode position-specific nucleotide
                position_vector = self.hdc.encode_nucleotide(nucleotide, i)
                
                # Combine sequence context with position-specific information
                final_vector = self.hdc.bind(encoded_seq, position_vector)
                
                # Predict coordinates
                predicted_coords = self.coordinate_predictor(final_vector.unsqueeze(0))
                coordinates.append(predicted_coords[0].cpu().numpy())
        
        return np.array(coordinates)
    
    def save_model(self, path):
        """Save the model to disk."""
        torch.save({
            'coordinate_predictor': self.coordinate_predictor.state_dict(),
            'ss_predictor': self.ss_predictor.state_dict(),
            'hdc_dimensions': self.hdc.dimensions
        }, path)
    
    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.coordinate_predictor.load_state_dict(checkpoint['coordinate_predictor'])
        self.ss_predictor.load_state_dict(checkpoint['ss_predictor'])
        
        # Recreate HDC if dimensions differ
        if checkpoint['hdc_dimensions'] != self.hdc.dimensions:
            self.hdc = HDC(dimensions=checkpoint['hdc_dimensions'], device=self.device) 