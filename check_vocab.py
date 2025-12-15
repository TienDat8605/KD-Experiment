from transformers import AutoTokenizer

def check_compatibility(model1, model2):
    print(f"Checking {model1} vs {model2}")
    try:
        tok1 = AutoTokenizer.from_pretrained(model1, trust_remote_code=True)
        tok2 = AutoTokenizer.from_pretrained(model2, trust_remote_code=True)
        
        print(f"Vocab size 1: {tok1.vocab_size}")
        print(f"Vocab size 2: {tok2.vocab_size}")
        
        if tok1.get_vocab() == tok2.get_vocab():
            print("Vocabularies are IDENTICAL. Logit KD is straightforward.")
        else:
            print("Vocabularies are DIFFERENT. Logit KD requires mapping or is not directly possible.")
            
    except Exception as e:
        print(f"Error loading tokenizers: {e}")

if __name__ == "__main__":
    check_compatibility("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
    print("-" * 20)
    check_compatibility("Qwen/Qwen2.5-7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct")
