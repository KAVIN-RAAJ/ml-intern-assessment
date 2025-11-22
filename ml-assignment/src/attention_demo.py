import numpy as np
from src.attention import scaled_dot_product_attention

def main():
    print("--- Scaled Dot-Product Attention Demo ---\n")
    
    # Define dimensions
    batch_size = 2
    seq_len_q = 3
    seq_len_k = 4 # Can be different from q
    d_k = 8 # Dimension of keys/queries
    d_v = 8 # Dimension of values
    
    print(f"Dimensions: Batch={batch_size}, Seq_Q={seq_len_q}, Seq_K={seq_len_k}, d_k={d_k}, d_v={d_v}\n")
    
    # Create random input matrices
    np.random.seed(42)
    Q = np.random.rand(batch_size, seq_len_q, d_k)
    K = np.random.rand(batch_size, seq_len_k, d_k)
    V = np.random.rand(batch_size, seq_len_k, d_v)
    
    print("Query (Q) shape:", Q.shape)
    print("Key (K) shape:", K.shape)
    print("Value (V) shape:", V.shape)
    
    # Call attention function
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("\n--- Output ---")
    print("Output shape:", output.shape)
    print("Output (first batch):\n", output[0])
    
    print("\n--- Attention Weights ---")
    print("Weights shape:", weights.shape)
    print("Weights (first batch):\n", weights[0])
    
    # Verify rows sum to 1
    print("\n--- Verification ---")
    row_sums = np.sum(weights, axis=-1)
    print("Row sums (should be close to 1):\n", row_sums)
    
    # Test with Mask
    print("\n--- Testing with Mask ---")
    # Mask out the last key for all queries
    mask = np.ones((batch_size, seq_len_q, seq_len_k))
    mask[:, :, -1] = 0
    
    print("Mask shape:", mask.shape)
    print("Mask (first batch, first query):\n", mask[0, 0])
    
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print("\nMasked Weights (first batch):\n", weights_masked[0])
    print("Masked Row sums:\n", np.sum(weights_masked, axis=-1))

if __name__ == "__main__":
    main()
