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
        Normalize 3D coordinates to zero mean and unit variance.
        
        Args:
            coordinates_df (pd.DataFrame): DataFrame with 3D coordinates
            
        Returns:
            pd.DataFrame: Normalized coordinates
        """
        # Calculate mean and std for each coordinate
        coords = ['x_1', 'y_1', 'z_1']
        
        for coord in coords:
            mean = coordinates_df[coord].mean()
            std = coordinates_df[coord].std()
            
            coordinates_df[coord] = (coordinates_df[coord] - mean) / std
            
        return coordinates_df
    
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