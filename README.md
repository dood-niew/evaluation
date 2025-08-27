# LLM Evaluation Framework

A comprehensive evaluation framework for Large Language Models supporting multiple model types and evaluation modes.

## Supported Models

- **Hugging Face Models**: Local transformers models
- **OpenAI API**: GPT models via OpenAI API
- **vLLM API**: Self-hosted or remote vLLM inference servers

## Evaluation Modes

### 1. Instruction Tuned (IT) Mode (`--mode it`)
For chat/instruction-following models with conversation format.

### 2. Pretrain Mode (`--mode pt`)
For base/pretrained models without instruction formatting.
> **Note**: Currently only supports Hugging Face models with batch size 1.

## Thinking Mode Options

- `True`: Enable thinking mode (if supported by model)
- `False`: Disable thinking mode 
- `None`: Used for pretrain mode evaluation

## Usage Examples

### Hugging Face Models

#### IT Mode with Thinking Enabled
```bash
python -m src.run \
  --model-path Qwen/Qwen3-8B \
  --batch-size 32 \
  --debug \
  -t True \
  --mode it
```

#### IT Mode with Thinking Disabled  
```bash
python -m src.run \
  --model-path Qwen/Qwen3-8B \
  --batch-size 32 \
  --debug \
  -t False \
  --mode it
```

#### Pretrain Mode
```bash
python -m src.run \
  --model-path Qwen/Qwen3-8B \
  --batch-size 1 \
  --debug \
  -t None \
  --mode pt
```

### OpenAI API
```bash
python -m src.run \
  --model-path gpt-4o-mini \
  --mode it \
  -t False \
  --debug \
  --batch-size 10 \
  --openai <YOUR_OPENAI_API_KEY>
```

### vLLM API
```bash
python -m src.run \
  --model-path thaillm-8b \
  --mode it \
  -t True \
  --debug \
  --batch-size 10 \
  --vllm http://example-host/example-sub/
```

## Available Arguments

### Required Arguments
- `--model-path`: Model name or path to evaluate
- `-t, --thinking-mode`: Enable/disable thinking mode (`True`/`False`/`None`)

### Model Source Arguments (choose one)
- `--openai`: OpenAI API key for using OpenAI models
- `--vllm`: Base URL for vLLM inference server
- Neither: Use local Hugging Face model

### Evaluation Configuration
- `--mode`: Evaluation mode (`pt` for pretrain, `it` for instruction-tuned)
- `--data`: List of datasets to evaluate (space-separated)
  - Default: `["belebele", "xcopa", "mmlu", "mmlu_thai", "xnli", "m3exam", "thai_exam", "m6exam"]`
- `--num-shots`: Number of few-shot examples (default: 5)
- `--batch-size`: Batch size for evaluation (default: 1)
- `--max-seq-len`: Maximum sequence length (default: 8192)

### Output Configuration
- `--save-result-dir`: Directory to save results (default: `./results`)
- `--save-time`: Add timestamp to results directory
- `--debug`: Run in debug mode (evaluates only first 10 samples)

### Other Options
- `--seed`: Random seed (default: 47)
- `--api-key`: API key for vLLM service (if required)

## Datasets

The framework supports evaluation on multiple Thai and multilingual datasets:

- **belebele**: Reading comprehension
- **xcopa**: Causal reasoning  
- **mmlu**: Massive multitask language understanding
- **mmlu_thai**: Thai version of MMLU
- **xnli**: Cross-lingual natural language inference
- **m3exam**: Thai educational exams for grade 9
- **thai_exam**: Thai standardized tests
- **m6exam**: Thai educational exams for grade 12

## Output

Results are saved in JSON format under the specified results directory with the following structure:
```
results/
└── {model_name}_{timestamp}/
    └── summary/
        ├── belebele_report.json
        ├── xcopa_report.json
        └── ...
```

## Notes

- **Batch Size Limitations**: 
  - Pretrain mode (`pt`) only supports batch size 1
  - API models can use higher batch sizes for parallel processing
- **Thinking Mode**: Only applicable for instruction-tuned models (`it` mode)
- **vLLM Connection**: The framework automatically tests vLLM connection before evaluation
