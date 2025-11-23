import argparse
import json
import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os

def cleanup():
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def translate_text(text, model, tokenizer, device, src_lang="eng_Latn", tgt_lang="ceb_Latn"):
    """Translates a single string."""
    if not text:
        return ""
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # Translate
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=512
        )
    
    # Decode
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    # Cleanup tensors immediately
    del inputs
    del generated_tokens
    cleanup()
    
    return result

def get_similarity(text1, text2, sim_model):
    """Computes cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    embeddings = sim_model.encode([text1, text2], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    del embeddings
    cleanup()
    return score

def main():
    parser = argparse.ArgumentParser(description="Translate GSM8K to Cebuano with QA")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (deprecated, use --end)")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--output", type=str, default="gsm8k_cebuano.jsonl", help="Output file path")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity score to save (0.0 to save all)")
    args = parser.parse_args()

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Translation Model & Tokenizer
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading translation model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if device == "cuda":
        model = model.half() # Use fp16 for GPU
    model = model.to(device)

    # Load Similarity Model
    print("Loading similarity model: all-MiniLM-L6-v2...")
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    sim_model = sim_model.to(device)
    
    # Load Dataset (Streaming)
    print("Loading GSM8K dataset (streaming)...")
    dataset = load_dataset("gsm8k", "main", split="train", streaming=True)
    
    # Determine start/end
    start_index = args.start
    end_index = args.end
    
    # If limit is used, it overrides end
    if args.limit:
        end_index = start_index + args.limit

    # Auto-resume logic (only if start is 0, otherwise trust user)
    if start_index == 0 and os.path.exists(args.output):
        with open(args.output, "r") as f:
            lines = sum(1 for _ in f)
            if lines > 0:
                print(f"Output file exists with {lines} lines. Resuming from index {lines}...")
                start_index = lines

    print(f"Processing range: {start_index} to {end_index if end_index else 'End'}")
    print(f"Saving to {args.output}...")

    with open(args.output, "a", encoding="utf-8") as f:
        for i, example in tqdm(enumerate(dataset)):
            if i < start_index:
                continue
            
            if end_index is not None and i >= end_index:
                print(f"Reached end index {end_index}. Stopping.")
                break
                
            original_question = example["question"]
            original_answer = example["answer"]
            
            # 1. Translate to Cebuano
            ceb_question = translate_text(original_question, model, tokenizer, device, src_lang="eng_Latn", tgt_lang="ceb_Latn")
            ceb_answer = translate_text(original_answer, model, tokenizer, device, src_lang="eng_Latn", tgt_lang="ceb_Latn")
            
            # 2. Back-translate to English
            back_question = translate_text(ceb_question, model, tokenizer, device, src_lang="ceb_Latn", tgt_lang="eng_Latn")
            
            # 3. Compute Similarity
            similarity = get_similarity(original_question, back_question, sim_model)
            
            # 4. Filter
            if similarity < args.threshold:
                print(f"Skipping index {i}: Similarity {similarity:.4f} < {args.threshold}")
                continue

            # Write to file
            output_data = {
                "original_question": original_question,
                "original_answer": original_answer,
                "cebuano_question": ceb_question,
                "cebuano_answer": ceb_answer,
                "back_translation": back_question,
                "similarity_score": similarity
            }
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            f.flush()
            
            cleanup()

    print("Translation complete.")

if __name__ == "__main__":
    main()
