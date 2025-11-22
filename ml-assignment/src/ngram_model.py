import random
from collections import defaultdict
from src.utils import clean_text, tokenize

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # Dictionary to store counts: counts[(w1, w2)][w3] = count
        self.counts = defaultdict(lambda: defaultdict(int))
        # Dictionary to store total counts for a context: total_counts[(w1, w2)] = total_count
        self.total_counts = defaultdict(int)

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # 1. Clean the text
        cleaned_text = clean_text(text)
        
        # 2. Split into sentences (assuming '.' is the delimiter from clean_text)
        # We remove empty strings that might result from split
        sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]

        for sentence in sentences:
            # 3. Tokenize
            tokens = tokenize(sentence)
            
            # 4. Pad with start and end tokens
            # For trigram, we need 2 start tokens to predict the first real word
            padded_tokens = ['<START>', '<START>'] + tokens + ['<END>']
            
            # 5. Count trigrams
            for i in range(len(padded_tokens) - 2):
                w1 = padded_tokens[i]
                w2 = padded_tokens[i+1]
                w3 = padded_tokens[i+2]
                
                self.counts[(w1, w2)][w3] += 1
                self.total_counts[(w1, w2)] += 1

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        # 1. Start with start tokens
        current_context = ('<START>', '<START>')
        generated_words = []
        
        for _ in range(max_length):
            # 2. Probabilistically choose the next word
            possible_next_words = self.counts[current_context]
            
            if not possible_next_words:
                # No continuation found (should not happen if trained well, but possible)
                break
                
            words = list(possible_next_words.keys())
            counts = list(possible_next_words.values())
            total = self.total_counts[current_context]
            
            # Calculate probabilities
            probs = [c / total for c in counts]
            
            # Sample
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            if next_word == '<END>':
                break
                
            generated_words.append(next_word)
            
            # 3. Update context
            current_context = (current_context[1], next_word)
            
        return ' '.join(generated_words)
