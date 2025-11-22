from src.ngram_model import TrigramModel

def main():
    # Create a new TrigramModel
    model = TrigramModel()

    # Train the model on the example corpus
    # Using Alice in Wonderland if available, else fallback
    import os
    corpus_file = "data/alice.txt" if os.path.exists("data/alice.txt") else "data/example_corpus.txt"
    print(f"Training on {corpus_file}...")
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()
    model.fit(text)

    # Generate new text
    generated_text = model.generate(max_length=100)
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
