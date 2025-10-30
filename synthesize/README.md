# WebArena Skill Synthesis

This directory contains all tools and scripts for synthesizing reusable Playwright skills from WebArena execution traces.

## Files

### Core Scripts

- **`synthesize_skills.py`** - Main synthesis script
  - Analyzes successful trajectories from WebArena results
  - Uses GPT-4o vision to synthesize Playwright functions
  - Outputs: `synthesized_skills.json`

- **`extract_skills_to_py.py`** - Skill extraction script
  - Converts JSON output to Python files
  - Organizes skills by site for `run.py` integration
  - Supports both single-file and directory outputs

### Configuration

- **`config.yaml`** - Synthesis configuration
  - Model settings (gpt-4o by default)
  - Synthesis parameters (min/max examples)
  - System prompt for skill generation

- **`environment-skills.yml`** - Conda environment definition
  - Python 3.11
  - OpenAI >= 1.0 (for vision support)
  - PyYAML, BeautifulSoup4, lxml

### Test & Examples

- **`test_json_synthesis.py`** - Test script for JSON-based synthesis
- **`skills.json`** - Example synthesized skills output

## Quick Start

### 1. Setup Environment

```bash
# Create the synthesis environment
conda env create -f environment-skills.yml
conda activate webarena-skills
```

### 2. Synthesize Skills

```bash
# Run synthesis on WebArena results
python synthesize_skills.py ../results/gpt-4-turbo-2024-04-09
```

This will:
- Convert HTML traces to JSON
- Group trajectories by site
- Send to GPT-4o for skill synthesis
- Output: `synthesized_skills.json`

### 3. Extract to Python

```bash
# Convert to directory structure for run.py
cd ..
python synthesize/extract_skills_to_py.py synthesize/synthesized_skills.json -d skills/
```

This creates:
```
skills/
├── shopping_admin/
│   ├── __init__.py
│   └── search_reviews.py
├── reddit/
│   └── create_post.py
└── ...
```

### 4. Use in run.py

```bash
# Run WebArena with synthesized skills
python run.py --skills_dir skills/ --model gpt-4 ...
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: gpt-4o  # Model for synthesis
  temperature: 0.7
  max_tokens: 4000

synthesis:
  max_examples_per_site: 10  # Max trajectories to analyze
  min_examples_required: 3   # Min successful examples needed

system_prompt: |
  [Instructions for skill generation]
```

## Environment

This synthesis pipeline uses a separate conda environment (`webarena-skills`) with:
- **OpenAI >= 1.0**: New API with vision support for GPT-4o
- **PyYAML**: For config file parsing

The main WebArena environment uses `openai==0.27.0` for compatibility with the core evaluation code.

## Workflow

```
WebArena Results → synthesize_skills.py → JSON → extract_skills_to_py.py → skills/ → run.py
       ↓                    ↓                              ↓
  render_*.html     Trajectory Analysis           Python Functions
   + images         (GPT-4o Vision)               Organized by Site
```

## Advanced Usage

### Custom Config

```bash
python synthesize_skills.py ../results/my-run --config my_config.yaml
```

### Single File Output

```bash
python extract_skills_to_py.py synthesized_skills.json --output all_skills.py
```

### Testing

```bash
# Test JSON synthesis pipeline
python test_json_synthesis.py
```

## Troubleshooting

### No skills synthesized
- Check: `min_examples_required` vs actual successful trajectories
- Lower threshold in `config.yaml`
- Verify `OPENAI_API_KEY` is set

### Import errors
- Ensure `conda activate webarena-skills` before running synthesis
- Check that `openai>=1.0` is installed

### Vision API errors
- GPT-4o required for vision (processes screenshots)
- Fallback: Edit `config.yaml` to use text-only model

## See Also

- `../SKILLS_WORKFLOW.md` - Complete end-to-end workflow
- `../EXTRACT_SKILLS_README.md` - Extract script details
- `CONFIG_UPDATE.md` - Configuration system documentation


