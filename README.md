# History QA Dataset Generator

Generate QA datasets from Vietnamese history books using Groq Cloud API.

## Features

- 🎯 **Mixed Question Types**: Simple Q&A, Multiple Choice, Fill-in-the-blank
- 🇻🇳 **Vietnamese Language**: All questions and answers in Vietnamese
- ✅ **Answer Rationale**: Each answer includes explanation/reasoning for verification
- 📚 **Smart Chunking**: Intelligent text segmentation for context preservation
- ⚡ **Rate Limiting**: Built-in delays to respect API limits
- 📄 **JSONL Output**: Easy-to-process JSON Lines format

## Setup

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or: venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file with your Groq API key:

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

Or export it directly:

```bash
export GROQ_API_KEY="your_api_key_here"
```

## Usage

### Basic Usage

```bash
python generate_qa.py
```

This will:
- Read all `.txt` files from the `data/` directory
- Generate ~500 QA pairs per book
- Save output to `output/` directory

### Advanced Options

```bash
python generate_qa.py \
    --input-dir data \
    --output-dir output \
    --target-qa 500 \
    --model llama-3.3-70b-versatile \
    --chunk-size 2000 \
    --delay 1.0
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input-dir` | `-i` | `data` | Input directory with .txt files |
| `--output-dir` | `-o` | `output` | Output directory for JSONL files |
| `--target-qa` | `-n` | `500` | Target QA pairs per book |
| `--model` | `-m` | `openai/gpt-oss-120b` | Groq model to use |
| `--chunk-size` | `-c` | `2000` | Text chunk size (chars) |
| `--delay` | `-d` | `1.0` | Delay between API calls (secs) |
| `--api-key` | `-k` | - | Groq API key (optional if set in env) |

## Output Format

The output is in JSONL format. Each line is a JSON object with a **rationale** field for answer verification:

### Simple Q&A
```json
{
  "question": "Ai là Toàn quyền Đông Dương năm 1897?",
  "answer": "Paul Doumer",
  "rationale": "Theo đoạn văn, tháng 2-1897, Paul Doumer được bổ nhiệm làm Toàn quyền Đông Dương.",
  "question_type": "simple_qa",
  "source_book": "lsvn07",
  "chunk_id": 5
}
```

### Multiple Choice
```json
{
  "question": "Năm nào Pháp bắt đầu khai thác thuộc địa lần thứ nhất?",
  "answer": "A",
  "rationale": "Đoạn văn nêu rõ cuộc khai thác thuộc địa lần thứ nhất kéo dài từ năm 1897, do đó đáp án A là chính xác.",
  "question_type": "multiple_choice",
  "choices": ["A. 1897", "B. 1900", "C. 1885", "D. 1910"],
  "source_book": "lsvn07",
  "chunk_id": 3
}
```

### Fill-in-the-blank
```json
{
  "question": "Cuộc khai thác thuộc địa lần thứ nhất kéo dài từ năm 1897 đến hết ___.",
  "answer": "Chiến tranh thế giới lần thứ nhất",
  "rationale": "Theo văn bản, cuộc khai thác đại quy mô này kéo dài từ năm 1897 đến hết Chiến tranh thế giới lần thứ nhất.",
  "question_type": "fill_in_blank",
  "source_book": "lsvn07",
  "chunk_id": 8
}
```

## Available Groq Models

### Text Generation Models

| Model ID | Description | Best For |
|----------|-------------|----------|
| `llama-3.3-70b-versatile` | Meta Llama 3.3 70B | ⭐ **Recommended** - Best quality |
| `llama-3.1-70b-versatile` | Meta Llama 3.1 70B | High quality, being deprecated |
| `llama-3.1-8b-instant` | Meta Llama 3.1 8B | Fast, lower quality |
| `llama-3-groq-70b-tool-use` | Llama 3 70B for tools | Function calling |
| `llama-3-groq-8b-tool-use` | Llama 3 8B for tools | Fast function calling |
| `mixtral-8x7b-32768` | Mistral Mixtral 8x7B | Good balance, 32K context |
| `gemma2-9b-it` | Google Gemma 2 9B | Fast, good quality |
| `deepseek-r1-distill-llama-70b` | DeepSeek R1 70B | Reasoning tasks |
| `qwen/qwen3-32b` | Alibaba Qwen3 32B | Multilingual |

### Newer Models (December 2024+)

| Model ID | Description |
|----------|-------------|
| `llama-4-scout` | Meta Llama 4 Scout |
| `llama-4-maverick` | Meta Llama 4 Maverick |
| `openai/gpt-oss-120b` | OpenAI GPT-OSS 120B |
| `openai/gpt-oss-20b` | OpenAI GPT-OSS 20B |
| `moonshotai/kimi-k2-instruct-0905` | Moonshot AI Kimi K2 |

> **Note**: Model availability may change. Check [console.groq.com](https://console.groq.com) for the latest list.

## Tips

1. **Quality vs Speed**: Use larger models (70B) for better quality, smaller models (8B) for speed
2. **Rate Limits**: Increase `--delay` if you hit rate limits
3. **Chunk Size**: Larger chunks provide more context but may exceed token limits
4. **Adding More Books**: Simply add `.txt` files to the `data/` directory
5. **Verify Rationale**: Use the `rationale` field to verify answer accuracy

## Data Directory Structure

```
HistoryQA/
├── venv/                 # Python virtual environment
├── data/
│   ├── lsvn07.txt        # Vietnamese History Vol 7 (1897-1918)
│   └── lsvn13.txt        # Vietnamese History Vol 13 (1965-1975)
├── output/
│   ├── lsvn07_qa.jsonl
│   └── lsvn13_qa.jsonl
├── generate_qa.py
├── requirements.txt
├── .env
└── README.md
```

## License

MIT
