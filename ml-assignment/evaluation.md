# Evaluation and Design Choices

## Task 1: N-Gram Language Model

### Design Choices

1.  **Data Structure**:
    *   I used a nested dictionary `defaultdict(lambda: defaultdict(int))` to store the trigram counts. This allows for efficient O(1) access to counts given a context `(w1, w2)` and a next word `w3`.
    *   I also maintained a `total_counts` dictionary `defaultdict(int)` to store the sum of counts for each context. This avoids re-summing the counts every time we need to calculate probabilities, improving generation speed.

2.  **Text Cleaning and Tokenization**:
    *   **Cleaning**: I implemented a simple cleaning strategy that converts text to lowercase and removes non-alphanumeric characters (except spaces and periods). This simplifies the vocabulary and reduces sparsity. I kept periods to allow for sentence splitting.
    *   **Tokenization**: I split the text into sentences based on periods, and then tokenized each sentence by splitting on whitespace. This ensures that n-grams do not cross sentence boundaries inappropriately.

3.  **Padding**:
    *   I padded each sentence with two `<START>` tokens and one `<END>` token.
    *   Two `<START>` tokens are necessary for a trigram model so that we can predict the first word of the sentence using the context `('<START>', '<START>')`.
    *   The `<END>` token allows the model to learn when to terminate a sentence.

4.  **Generation**:
    *   The generation process starts with the `('<START>', '<START>')` context.
    *   At each step, I calculate the probability distribution over the vocabulary given the current context.
    *   I use `random.choices` to sample the next word based on these probabilities. This ensures that the generation is probabilistic and diverse, rather than deterministic (always picking the most likely word).
    *   Generation stops when the `<END>` token is sampled or the maximum length is reached.

### Trade-offs and Future Improvements
*   **Sparsity**: Trigram models suffer from data sparsity. If a context was not seen during training, the model cannot generate a continuation. I implemented a basic check to stop if no continuation is found. Smoothing techniques (like Laplace smoothing or Kneser-Ney) could be added to handle unseen n-grams.
*   **Vocabulary Size**: I did not limit the vocabulary size. For very large corpora, this could lead to memory issues. Implementing a frequency cutoff and replacing rare words with `<UNK>` would be a good improvement.

## Task 2: Scaled Dot-Product Attention

### Implementation Explanation

I implemented the Scaled Dot-Product Attention mechanism using only `numpy` as requested. The function `scaled_dot_product_attention(Q, K, V, mask=None)` follows these steps:

1.  **Score Calculation**:
    *   We compute the dot product of the Query matrix `Q` and the Transpose of the Key matrix `K^T`.
    *   `Q` has shape `(..., seq_len_q, d_k)` and `K` has shape `(..., seq_len_k, d_k)`.
    *   The result is a score matrix of shape `(..., seq_len_q, seq_len_k)`.

2.  **Scaling**:
    *   The scores are divided by `sqrt(d_k)`. This scaling factor prevents the dot products from growing too large in magnitude, which would push the softmax function into regions with extremely small gradients.

3.  **Masking (Optional)**:
    *   If a mask is provided, we apply it to the scores before the softmax.
    *   Positions to be masked (where mask is 0 or False) are set to a very large negative number (e.g., `-1e9`).
    *   This ensures that after applying softmax, the attention weight for these positions becomes effectively zero.

4.  **Softmax**:
    *   We apply the softmax function along the last dimension (key sequence length) to obtain the attention weights.
    *   I implemented a numerically stable softmax by subtracting the maximum value from the logits before exponentiation.

5.  **Weighted Sum**:
    *   Finally, we compute the dot product of the attention weights and the Value matrix `V`.
    *   This produces the final output of shape `(..., seq_len_q, d_v)`.

### Demonstration
A demonstration script is provided in `src/attention_demo.py`. It creates random Q, K, V matrices and verifies:
*   The output shapes are correct.
*   The attention weights sum to 1.
*   Masking correctly sets weights to 0 for masked positions.


