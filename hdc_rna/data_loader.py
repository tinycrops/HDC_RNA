import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class RNADataLoader:
    """
    Data loader for RNA 3D folding dataset.
    """
    
    def __init__(self, data_dir):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the directory containing RNA data files
        """
        self.data_dir = data_dir
        
    def load_train_data(self):
        """
        Load training data.
        
        Returns:
            tuple: (sequences_df, coordinates_df)
        """
        seq_path = os.path.join(self.data_dir, 'train_sequences.csv')
        labels_path = os.path.join(self.data_dir, 'train_labels.csv')
        
        sequences_df = pd.read_csv(seq_path)
        coordinates_df = pd.read_csv(labels_path)
        
        return sequences_df, coordinates_df
    
    def load_validation_data(self):
        """
        Load validation data.
        
        Returns:
            tuple: (sequences_df, coordinates_df)
        """
        seq_path = os.path.join(self.data_dir, 'validation_sequences.csv')
        labels_path = os.path.join(self.data_dir, 'validation_labels.csv')
        
        sequences_df = pd.read_csv(seq_path)
        coordinates_df = pd.read_csv(labels_path)
        
        return sequences_df, coordinates_df
    
    def load_test_data(self):
        """
        Load test data (sequences only, no coordinates).
        
        Returns:
            pd.DataFrame: Test sequences
        """
        test_path = os.path.join(self.data_dir, 'test_sequences.csv')
        return pd.read_csv(test_path)
    
    def get_sample_data(self, max_sequences=10):
        """
        Get a small sample of data for quick testing.
        
        Args:
            max_sequences (int): Maximum number of sequences to include
            
        Returns:
            tuple: (sequences_df, coordinates_df)
        """
        sequences_df, coordinates_df = self.load_train_data()
        
        # Take a small sample of sequences
        sample_sequences = sequences_df.head(max_sequences)
        
        # Filter coordinates for these sequences
        sample_ids = sample_sequences['target_id'].tolist()
        sample_coords = coordinates_df[coordinates_df['ID'].apply(
            lambda x: any(x.startswith(seq_id) for seq_id in sample_ids)
        )]
        
        return sample_sequences, sample_coords
    
    def preprocess_sequences(self, sequences_df):
        """
        Preprocess RNA sequences.
        
        Args:
            sequences_df (pd.DataFrame): DataFrame with RNA sequences
            
        Returns:
            pd.DataFrame: Preprocessed sequences
        """
        # Replace T with U (in case of DNA sequences)
        sequences_df['sequence'] = sequences_df['sequence'].str.replace('T', 'U')
        
        # Filter out sequences with non-standard nucleotides
        sequences_df = sequences_df[sequences_df['sequence'].str.match('^[ACGU]+$')]
        
        return sequences_df
    
    def normalize_coordinates(self, coordinates_df):
        """
        Normalize 3D coordinates using robust scaling methods.
        
        Args:
            coordinates_df (pd.DataFrame): DataFrame with 3D coordinates
            
        Returns:
            pd.DataFrame: Normalized coordinates
        """
        # Calculate robust normalization parameters
        coords = ['x_1', 'y_1', 'z_1']
        
        # Create a copy to avoid changing the original
        normalized_df = coordinates_df.copy()
        
        # Get statistics for each coordinate
        stats = {}
        for coord in coords:
            # Use percentiles for more robust scaling
            q1 = coordinates_df[coord].quantile(0.05)
            q3 = coordinates_df[coord].quantile(0.95)
            iqr = q3 - q1
            
            # Handle extreme outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Clip values to remove extreme outliers
            normalized_df[coord] = normalized_df[coord].clip(lower_bound, upper_bound)
            
            # Calculate mean and standard deviation after clipping
            mean = normalized_df[coord].mean()
            std = normalized_df[coord].std()
            
            # Ensure non-zero std to prevent division by zero
            if std < 1e-6:
                std = 1.0
                
            # Store stats for logging
            stats[coord] = {
                'mean': mean,
                'std': std,
                'min': normalized_df[coord].min(),
                'max': normalized_df[coord].max()
            }
            
            # Apply z-score normalization
            normalized_df[coord] = (normalized_df[coord] - mean) / std
            
        # Log statistics
        print("Coordinate normalization statistics:")
        for coord, stat in stats.items():
            print(f"  {coord}: mean={stat['mean']:.3f}, std={stat['std']:.3f}, range=[{stat['min']:.3f}, {stat['max']:.3f}]")
            
        return normalized_df
    
    def prepare_training_data(self, max_sequences=None, normalize=True):
        """
        Prepare data for training.
        
        Args:
            max_sequences (int, optional): Maximum number of sequences to use
            normalize (bool): Whether to normalize coordinates
            
        Returns:
            tuple: (sequences_df, coordinates_df)
        """
        # Load and preprocess training data
        sequences_df, coordinates_df = self.load_train_data()
        sequences_df = self.preprocess_sequences(sequences_df)
        
        # Limit to max_sequences if specified
        if max_sequences is not None:
            sequences_df = sequences_df.head(max_sequences)
            
            # Filter coordinates for these sequences
            sample_ids = sequences_df['target_id'].tolist()
            coordinates_df = coordinates_df[coordinates_df['ID'].apply(
                lambda x: any(x.startswith(seq_id) for seq_id in sample_ids)
            )]
        
        # Normalize coordinates if needed
        if normalize:
            coordinates_df = self.normalize_coordinates(coordinates_df)
            
        return sequences_df, coordinates_df 