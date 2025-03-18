import numpy as np
import torch

class HDC:
    """
    Hyperdimensional Computing utilities for RNA folding prediction.
    """
    def __init__(self, dimensions=10000, device='cpu'):
        """
        Initialize the HDC system with specified dimensions.
        
        Args:
            dimensions (int): Dimensionality of hypervectors
            device (str): Device to use for torch operations ('cpu' or 'cuda')
        """
        self.dimensions = dimensions
        self.device = device
        
        # Define base hypervectors for nucleotides
        self.base_vectors = {
            'A': self._random_hvector(),
            'C': self._random_hvector(),
            'G': self._random_hvector(),
            'U': self._random_hvector()
        }
        
        # Initialize position hypervectors
        self.position_vectors = {}
        
    def _random_hvector(self):
        """Generate a random binary hypervector."""
        # Create bipolar vector with elements {-1, 1}
        hvector = 2 * torch.bernoulli(torch.ones(self.dimensions) * 0.5) - 1
        return hvector.to(self.device)
    
    def similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two hypervectors.
        
        Args:
            vec1, vec2: Hypervectors to compare
            
        Returns:
            float: Cosine similarity (between -1 and 1)
        """
        return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
    
    def bind(self, vec1, vec2):
        """
        Binding operation (element-wise multiplication).
        Used to bind features to values or create associations.
        
        Args:
            vec1, vec2: Hypervectors to bind
            
        Returns:
            torch.Tensor: Result of binding
        """
        return vec1 * vec2
    
    def bundle(self, vectors):
        """
        Bundling operation (vector addition).
        Used to combine multiple hypervectors representing a set.
        
        Args:
            vectors: List of hypervectors to bundle
            
        Returns:
            torch.Tensor: Result of bundling
        """
        if not vectors:
            return None
            
        result = torch.zeros_like(vectors[0])
        for vec in vectors:
            result += vec
            
        # Apply a threshold to convert back to binary or bipolar
        # result = torch.sign(result)  # For bipolar {-1, 1}
        
        return result
    
    def permute(self, vector, shift=1):
        """
        Permutation operation (cyclic shift).
        Used to represent sequence order.
        
        Args:
            vector: Hypervector to permute
            shift: Number of positions to shift
            
        Returns:
            torch.Tensor: Permuted hypervector
        """
        return torch.roll(vector, shifts=shift)
    
    def get_position_vector(self, position):
        """
        Get a position-specific hypervector, generating if needed.
        
        Args:
            position: Position in the sequence
            
        Returns:
            torch.Tensor: Position hypervector
        """
        if position not in self.position_vectors:
            self.position_vectors[position] = self._random_hvector()
        
        return self.position_vectors[position]
    
    def encode_nucleotide(self, nucleotide, position):
        """
        Encode a nucleotide at a specific position.
        
        Args:
            nucleotide: One of 'A', 'C', 'G', 'U'
            position: Position in the RNA sequence
            
        Returns:
            torch.Tensor: Encoded hypervector
        """
        if nucleotide not in self.base_vectors:
            raise ValueError(f"Unknown nucleotide: {nucleotide}")
            
        nucleotide_vector = self.base_vectors[nucleotide]
        position_vector = self.get_position_vector(position)
        
        # Bind nucleotide with its position
        return self.bind(nucleotide_vector, position_vector)
    
    def encode_sequence(self, sequence):
        """
        Encode a complete RNA sequence.
        
        Args:
            sequence: RNA sequence string (e.g., "GGGAACCC")
            
        Returns:
            torch.Tensor: Hypervector representing the sequence
        """
        vectors = []
        
        for i, nucleotide in enumerate(sequence):
            vectors.append(self.encode_nucleotide(nucleotide, i))
            
        # Bundle all position-encoded nucleotides
        return self.bundle(vectors)
    
    def encode_sequence_with_ngrams(self, sequence, n=3):
        """
        Encode a sequence using n-grams to capture local context.
        
        Args:
            sequence: RNA sequence string
            n: Size of n-grams
            
        Returns:
            torch.Tensor: Hypervector representing the sequence with n-gram context
        """
        if len(sequence) < n:
            return self.encode_sequence(sequence)
            
        vectors = []
        
        # Process each n-gram
        for i in range(len(sequence) - n + 1):
            ngram = sequence[i:i+n]
            
            # First, create a bundled representation of the n-gram
            ngram_vectors = [self.base_vectors[nucleotide] for nucleotide in ngram]
            ngram_vector = self.bundle(ngram_vectors)
            
            # Then bind it with position information
            position_vector = self.get_position_vector(i)
            vectors.append(self.bind(ngram_vector, position_vector))
            
        return self.bundle(vectors) 