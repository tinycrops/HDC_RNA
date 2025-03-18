import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hdc_utils import HDC

def test_basic_operations():
    """Test basic HDC operations."""
    print("Testing basic HDC operations...")
    
    # Initialize HDC with smaller dimensions for testing
    hdc = HDC(dimensions=1000)
    
    # Test random vector generation
    vec1 = hdc._random_hvector()
    vec2 = hdc._random_hvector()
    
    print(f"Vector shapes: {vec1.shape}, {vec2.shape}")
    
    # Test binding
    bound = hdc.bind(vec1, vec2)
    print(f"Binding result shape: {bound.shape}")
    
    # Test unbinding (binding with same vector should recover the other)
    # For bipolar vectors: x * y * y = x (since y * y = 1)
    recovered = hdc.bind(bound, vec2)
    similarity = hdc.similarity(recovered, vec1)
    print(f"Similarity after unbinding: {similarity:.4f} (should be close to 1.0)")
    
    # Test bundling
    bundle = hdc.bundle([vec1, vec2])
    print(f"Bundling result shape: {bundle.shape}")
    
    # Test permutation
    permuted = hdc.permute(vec1)
    similarity = hdc.similarity(permuted, vec1)
    print(f"Similarity after permutation: {similarity:.4f} (should be < 0.1)")
    
    # Test sequence encoding
    sequence = "GGGACUGACGAUCACGCAGUCUAU"
    encoded = hdc.encode_sequence(sequence)
    print(f"Encoded sequence shape: {encoded.shape}")
    
    # Test n-gram encoding
    encoded_ngram = hdc.encode_sequence_with_ngrams(sequence, n=3)
    print(f"N-gram encoded sequence shape: {encoded_ngram.shape}")
    
    print("Basic operations test completed.")

def test_similarity_properties():
    """Test similarity properties of HDC vectors."""
    print("\nTesting similarity properties...")
    
    # Initialize HDC with higher dimensions for better statistics
    hdc = HDC(dimensions=10000)
    
    # Generate multiple base vectors
    num_vectors = 10
    vectors = [hdc._random_hvector() for _ in range(num_vectors)]
    
    # Test orthogonality of random vectors
    similarities = []
    for i in range(num_vectors):
        for j in range(i+1, num_vectors):
            similarities.append(hdc.similarity(vectors[i], vectors[j]))
    
    avg_similarity = np.mean([s.item() for s in similarities])
    std_similarity = np.std([s.item() for s in similarities])
    
    print(f"Average similarity between random vectors: {avg_similarity:.4f} ± {std_similarity:.4f}")
    print(f"Should be close to 0 with standard deviation around 1/sqrt(D) = {1/np.sqrt(hdc.dimensions):.4f}")
    
    # Test similarity with noisy vectors
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    orig_vector = hdc._random_hvector()
    
    for noise in noise_levels:
        # Create noisy vector by flipping bits
        noise_mask = torch.bernoulli(torch.ones(hdc.dimensions) * noise)
        noisy_vector = orig_vector * (1 - 2 * noise_mask)  # Flip signs where noise_mask is 1
        
        sim = hdc.similarity(orig_vector, noisy_vector)
        print(f"Similarity with {noise*100:.1f}% noise: {sim:.4f}")

def test_rna_encoding():
    """Test RNA-specific encoding."""
    print("\nTesting RNA encoding...")
    
    # Initialize HDC
    hdc = HDC(dimensions=5000)
    
    # Test sequences of different lengths
    sequences = [
        "GGGACU",
        "GGGACUGACGAU",
        "GGGACUGACGAUCACGCAGUCUAU"
    ]
    
    for seq in sequences:
        # Encode sequence
        encoded = hdc.encode_sequence(seq)
        
        # Encode with n-grams
        encoded_ngram = hdc.encode_sequence_with_ngrams(seq, n=3)
        
        # Compare similarity
        sim = hdc.similarity(encoded, encoded_ngram)
        
        print(f"Sequence '{seq}' (length {len(seq)}):")
        print(f"  Regular vs N-gram encoding similarity: {sim:.4f}")
        
        # Test similarity with mutations
        # Mutate a random position
        mutated_seq = list(seq)
        pos = np.random.randint(0, len(seq))
        current = mutated_seq[pos]
        choices = [n for n in "ACGU" if n != current]
        mutated_seq[pos] = np.random.choice(choices)
        mutated_seq = "".join(mutated_seq)
        
        # Encode mutated sequence
        encoded_mutated = hdc.encode_sequence(mutated_seq)
        
        # Compare similarity
        sim = hdc.similarity(encoded, encoded_mutated)
        
        print(f"  Similarity after mutation at position {pos}: {sim:.4f}")
        print()

def analyze_sequence_similarities():
    """Analyze similarities between different RNA sequences from the dataset."""
    print("\nAnalyzing sequence similarities...")
    
    # Use a simple dataset of RNA sequences
    sequences = [
        "GGGUGCUCAGUACGAGAGGAACCGCACCC",  # 1SCL_A from the training data
        "GGCGCAGUGGGCUAGCGCCACUCAAAAGGCCCAU",  # 1RNK_A
        "GGGACUGACGAUCACGCAGUCUAU",  # 1RHT_A
        "GGGGGCCACAGCAGAAGCGUUCACGUCGCAGCCCCUGUCAGCCAUUGCACUCCGGCUGCGAAUUCUGCU"  # R1107
    ]
    
    # Initialize HDC
    hdc = HDC(dimensions=10000)
    
    # Encode all sequences
    encodings = [hdc.encode_sequence(seq) for seq in sequences]
    encodings_ngram = [hdc.encode_sequence_with_ngrams(seq, n=3) for seq in sequences]
    
    # Calculate pairwise similarities
    print("Regular encoding similarities:")
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            sim = hdc.similarity(encodings[i], encodings[j])
            print(f"  Seq{i+1} vs Seq{j+1}: {sim:.4f}")
    
    print("\nN-gram encoding similarities:")
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            sim = hdc.similarity(encodings_ngram[i], encodings_ngram[j])
            print(f"  Seq{i+1} vs Seq{j+1}: {sim:.4f}")

def main():
    """Run tests."""
    test_basic_operations()
    test_similarity_properties()
    test_rna_encoding()
    analyze_sequence_similarities()

if __name__ == "__main__":
    main() 