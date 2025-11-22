import re

def clean_text(text):
    """
    Cleans the text by converting to lowercase and removing non-alphanumeric characters (keeping spaces).
    """
    text = text.lower()
    # Keep only alphanumeric characters and spaces. 
    # You might want to keep punctuation for a better model, but for simplicity we remove it as per common basic n-gram tasks.
    # However, keeping sentence boundaries is crucial for n-grams. 
    # Let's keep periods to denote sentence ends if we want to be fancy, 
    # but the assignment hints say "padding the text with start and end tokens".
    # Usually this implies sentence-level processing.
    # Let's assume the input text is a full corpus and we might want to split by sentences first?
    # For this simple assignment, let's just clean everything to lowercase and remove special chars, 
    # treating the whole text as one long sequence or split by simple punctuation.
    
    # A robust way: replace newlines with spaces, remove weird chars.
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text) # Keep dots for sentence splitting potential
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """
    Tokenizes the text into words.
    """
    return text.split()

