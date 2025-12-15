import json
import argparse
from pathlib import Path
from typing import List, Dict, Generator, Tuple
import re
from tqdm import tqdm
from datasets import Dataset, DatasetDict

def clean_text(text: str) -> str:
    """Clean OCR artifacts and normalize text (Same as generate_qa.py)"""
    # Remove page numbers and headers
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^LỊCH SỬ VIỆT NAM.*$', '', text, flags=re.MULTILINE)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove OCR artifacts (common patterns)
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\"\'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]', '', text)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> Generator[Tuple[int, str], None, None]:
    """Split text into overlapping chunks (Same as generate_qa.py)"""
    paragraphs = text.split('\n\n')
    current_chunk = ""
    chunk_id = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            yield chunk_id, current_chunk.strip()
            chunk_id += 1
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
        
        current_chunk += "\n\n" + para
    
    if current_chunk.strip():
        yield chunk_id, current_chunk.strip()

def load_source_chunks(source_dir: Path) -> Dict[str, Dict[int, str]]:
    """Load all text files and split into chunks mapping: {book_name: {chunk_id: text}}"""
    source_map = {}
    files = list(source_dir.glob("*.txt"))
    
    for file_path in files:
        book_name = file_path.stem
        print(f"Loading source: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = clean_text(f.read())
            
        chunks = dict(chunk_text(text))
        source_map[book_name] = chunks
        
    return source_map

def main():
    parser = argparse.ArgumentParser(description="Prepare History QA Dataset")
    parser.add_argument("--source-dir", type=str, default="source")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="final_dataset")
    args = parser.parse_args()

    source_path = Path(args.source_dir)
    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    
    # 1. Load Sources
    sources = load_source_chunks(source_path)
    
    # 2. Load QA Pairs and Map Context
    data_samples = []
    
    qa_files = list(dataset_path.glob("*_qa.jsonl"))
    print(f"Found {len(qa_files)} QA files")
    
    missing_chunks = 0
    total_samples = 0
    
    for qa_file in qa_files:
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                total_samples += 1
                
                book = item.get('source_book')
                chunk_id = item.get('chunk_id')
                
                # Retrieve context
                context = sources.get(book, {}).get(chunk_id)
                
                if context:
                    # Create formatted sample
                    # Format: Context: ... \n Question: ... \n Answer: ...
                    data_samples.append({
                        "context": context,
                        "question": item['question'],
                        "answer": item['answer'],
                        "question_type": item['question_type'],
                        "rationale": item['rationale'],
                        "source": book,
                        "text": f"Context:\n{context}\n\nQuestion:\n{item['question']}\n\nAnswer:\n{item['answer']}"
                    })
                else:
                    missing_chunks += 1
                    
    print(f"Total processed: {len(data_samples)}/{total_samples} (Missing contexts: {missing_chunks})")
    
    # 3. Create Dataset and Split
    full_ds = Dataset.from_list(data_samples)
    
    # 75% Train, 10% Val, 15% Test
    # First split Test (15%)
    train_dev_test = full_ds.train_test_split(test_size=0.15, seed=42)
    test_ds = train_dev_test['test']
    train_dev = train_dev_test['train']
    
    # Then split Train/Val (10% of total is ~11.7% of remaining 85%)
    # 0.10 / 0.85 = 0.1176
    train_val = train_dev.train_test_split(test_size=0.1176, seed=42)
    
    final_ds = DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': test_ds
    })
    
    print("Dataset Split:")
    print(final_ds)
    
    # 4. Save
    final_ds.save_to_disk(args.output_dir)
    print(f"Saved dataset to {args.output_dir}")

if __name__ == "__main__":
    main()
