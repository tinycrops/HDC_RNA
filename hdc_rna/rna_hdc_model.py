import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .hdc_utils import HDC

class RNAHDC3DPredictor:
    """
    RNA 3D structure predictor using Hyperdimensional Computing.
    
    This class implements a model that predicts 3D coordinates of RNA nucleotides
    using Hyperdimensional Computing combined with neural networks.
    """
    
    def __init__(self, hdc_dimensions=5000, device='cpu'):
        """
        Initialize the HDC RNA 3D predictor.
        
        Args:
            hdc_dimensions (int): Dimensionality of hypervectors
            device (str): Device to use for training (cuda or cpu)
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self.device = device
        
        # Initialize HDC
        self.hdc = HDC(dimensions=hdc_dimensions, device=device)
        
        # Weight initialization function with smaller variance
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.1  # Scale down weights to prevent initial large gradients
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize coordinate predictor with more regularization and batch normalization
        self.coordinate_predictor = nn.Sequential(
            nn.Linear(hdc_dimensions, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # 3D coordinates
        ).to(device)
        
        # Apply weight initialization
        self.coordinate_predictor.apply(init_weights)
        
        # Initialize secondary structure predictor
        self.ss_predictor = nn.Sequential(
            nn.Linear(hdc_dimensions, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 secondary structure classes
        ).to(device)
        
        # Apply weight initialization to SS predictor
        self.ss_predictor.apply(init_weights)
        
        # Define loss functions
        self.coordinate_loss = nn.MSELoss(reduction='mean')
        self.ss_loss = nn.CrossEntropyLoss()
        
        # Initialize optimizer with gradient clipping
        self.optimizer = optim.AdamW(  # Switch to AdamW for better regularization
            list(self.coordinate_predictor.parameters()) + 
            list(self.ss_predictor.parameters()),
            lr=0.0003,  # Lower learning rate
            weight_decay=1e-4,
            eps=1e-8  # Higher epsilon for numerical stability
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True, 
            min_lr=1e-6, threshold=0.01
        )
    
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
        Prepare training data.
        
        Args:
            sequences_df (pd.DataFrame): RNA sequences
            coordinates_df (pd.DataFrame): 3D coordinates
            batch_size (int): Batch size for training
            
        Returns:
            DataLoader: Training data loader
        """
        from torch.utils.data import TensorDataset, DataLoader
        import torch
        from tqdm import tqdm
        
        data = []
        
        # Process each sequence
        for _, seq_row in tqdm(sequences_df.iterrows(), total=len(sequences_df), desc="Encoding sequences"):
            target_id = seq_row['target_id']
            sequence = seq_row['sequence']
            
            # Filter coordinates for this sequence
            seq_coords = coordinates_df[coordinates_df['ID'].str.startswith(target_id)]
                
            if len(seq_coords) == 0:
                continue
                
            # Encode the sequence on CPU to save GPU memory
            with torch.no_grad():
                encoded_seq = self.encode_rna_sequence(sequence).cpu()
            
            # Process nucleotides in chunks to reduce memory usage
            chunk_size = 100  # Process 100 nucleotides at a time
            for i in range(0, len(seq_coords), chunk_size):
                chunk_coords = seq_coords.iloc[i:i+chunk_size]
                
                # Process each nucleotide in the chunk
                for _, coord_row in chunk_coords.iterrows():
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
                        
                    # Process each nucleotide with no gradient tracking
                    with torch.no_grad():
                        position_vector = self.hdc.encode_nucleotide(nucleotide, position).cpu()
                        # Combine sequence context with position-specific information
                        final_vector = self.hdc.bind(encoded_seq, position_vector).cpu()
                    
                    data.append((final_vector, coords))
        
        # Create dataset in batches to save memory
        print("Creating dataset")
        if len(data) == 0:
            raise ValueError("No data to train on. Check that the sequence IDs match coordinate IDs.")
            
        try:
            X = torch.stack([item[0] for item in data])
            y = torch.stack([item[1] for item in data])
            
            # Create dataset and dataloader
            dataset = TensorDataset(X, y)
            
            # Clear memory
            data = None
            X = None
            y = None
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"Data preparation complete - created dataset with {len(dataset)} samples")
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("Out of memory error during data preparation. Try reducing HDC dimensions or batch size.")
                raise
            else:
                raise
    
    def train(self, train_loader, epochs=10, accumulate_grad=1):
        """
        Train the HDC RNA 3D predictor.
        
        Args:
            train_loader (DataLoader): Training data loader
            epochs (int): Number of training epochs
            accumulate_grad (int): Number of gradient accumulation steps
            
        Returns:
            list: Training losses
        """
        import torch
        import numpy as np
        from tqdm import tqdm
        
        losses = []
        nan_count = 0
        max_nan_threshold = 50  # Maximum number of NaN batches before reducing learning rate
        
        # Clear gradients at start
        self.optimizer.zero_grad()
        
        # Use amp (automatic mixed precision) to improve numerical stability
        if hasattr(torch.cuda, 'amp') and self.device != 'cpu':
            scaler = torch.cuda.amp.GradScaler()
            using_amp = True
        else:
            using_amp = False
            
        print(f"Using automatic mixed precision: {using_amp}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_idx = 0
            valid_batches = 0
            epoch_nan_count = 0
            
            # Apply gradient clipping 
            for param_group in self.optimizer.param_groups:
                if epoch == 0:
                    # Lower initial learning rate for first epoch
                    param_group['lr'] = param_group['lr'] * 0.1
                elif epoch == 1:
                    # Restore learning rate after first epoch
                    param_group['lr'] = param_group['lr'] * 10
            
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                try:
                    # Move data to device
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Forward pass with mixed precision
                    if using_amp:
                        with torch.cuda.amp.autocast():
                            # Forward pass
                            predicted_coords = self.coordinate_predictor(batch_x)
                            
                            # Compute loss
                            loss = self.coordinate_loss(predicted_coords, batch_y)
                            
                            # Check for NaN or Inf
                            if not torch.isfinite(loss).all():
                                raise ValueError("NaN or Inf loss detected")
                                
                            # Scale the loss to account for gradient accumulation
                            loss = loss / accumulate_grad
                        
                        # Scale loss and backward
                        scaler.scale(loss).backward()
                        
                        # Update weights only when we've accumulated enough gradients
                        if (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(train_loader):
                            # Unscale before clipping gradients
                            scaler.unscale_(self.optimizer)
                            
                            # Clip gradients to prevent exploding gradients (stricter clipping)
                            torch.nn.utils.clip_grad_norm_(
                                list(self.coordinate_predictor.parameters()) + 
                                list(self.ss_predictor.parameters()),
                                max_norm=0.5
                            )
                            
                            # Step with scaler
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        # Forward pass
                        predicted_coords = self.coordinate_predictor(batch_x)
                        
                        # Compute loss
                        loss = self.coordinate_loss(predicted_coords, batch_y)
                        
                        # Check for NaN or Inf
                        if not torch.isfinite(loss).all():
                            raise ValueError("NaN or Inf loss detected")
                        
                        # Scale the loss to account for gradient accumulation
                        loss = loss / accumulate_grad
                        
                        # Backpropagation
                        loss.backward()
                        
                        # Update weights only when we've accumulated enough gradients
                        if (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(train_loader):
                            # Clip gradients to prevent exploding gradients (stricter clipping)
                            torch.nn.utils.clip_grad_norm_(
                                list(self.coordinate_predictor.parameters()) + 
                                list(self.ss_predictor.parameters()),
                                max_norm=0.5
                            )
                            
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    
                    epoch_loss += loss.item() * accumulate_grad  # Scale back for reporting
                    valid_batches += 1
                    
                except (ValueError, RuntimeError) as e:
                    error_type = "NaN/Inf" if "NaN" in str(e) or "Inf" in str(e) else "CUDA"
                    
                    if error_type == "NaN/Inf":
                        epoch_nan_count += 1
                        nan_count += 1
                        print(f"| WARNING: {error_type} error at batch {batch_idx}, skipping. Total: {nan_count}")
                        
                        # Reduce learning rate if too many NaN errors
                        if epoch_nan_count > max_nan_threshold:
                            print(f"Too many NaN errors ({epoch_nan_count}). Reducing learning rate by 5x.")
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = max(param_group['lr'] * 0.2, 1e-7)
                                
                            # Reset counter
                            epoch_nan_count = 0
                            
                            # Move to next epoch if learning rate is very small
                            if param_group['lr'] <= 1e-6:
                                print("Learning rate too small, moving to next epoch.")
                                break
                    elif 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, skipping batch')
                    else:
                        print(f"| ERROR: {str(e)}")
                        raise e
                        
                    # Skip this batch and clear gradients
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                
                batch_idx += 1
            
            # Calculate average loss properly
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                losses.append(avg_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Valid Batches: {valid_batches}")
                
                # Update learning rate based on loss
                self.scheduler.step(avg_loss)
                
                # Print current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.8f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: No valid batches, skipping")
                losses.append(float('nan'))
            
            # Save model checkpoint after each epoch
            if (epoch + 1) % 2 == 0:  # Save more frequently
                checkpoint_path = f"models/rna_hdc_model_epoch_{epoch+1}.pt"
                self.save_model(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Stop if loss is NaN for 2 consecutive epochs
            if len(losses) >= 2 and np.isnan(losses[-1]) and np.isnan(losses[-2]):
                print("Loss is NaN for 2 consecutive epochs, stopping training")
                break
        
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