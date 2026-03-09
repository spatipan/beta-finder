# uv Dependency Management Guide

This project uses **uv**, a fast and reliable Python package manager written in Rust.

## Installation

If uv is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then add it to your PATH (zsh, bash):
```bash
source $HOME/.local/bin/env
```

Or for fish:
```bash
source $HOME/.local/bin/env.fish
```

Verify installation:
```bash
uv --version
```

## Quick Commands

### Initial Setup

```bash
# Create virtual environment and install dependencies
uv sync

# Create venv but skip installation (if you just want to set up structure)
uv venv
```

### Running Scripts

```bash
# Run Python directly in the project's venv
uv run python scrape.py
uv run python embed.py --model ViT-L-14 --pretrained openai
uv run python search.py photo.jpg --top 5

# Or activate venv manually
source .venv/bin/activate
python scrape.py
```

### Managing Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev pytest

# Update all dependencies
uv update

# Update specific package
uv update package-name

# View locked dependencies
cat uv.lock | head -50
```

### Virtual Environment

```bash
# Show venv location
uv venv --path

# Remove venv (if needed)
rm -rf .venv

# Recreate from lock file
uv sync
```

## Project Configuration

Dependencies are defined in `pyproject.toml`:

- **Main dependencies** — All production packages
- **Optional groups**:
  - `dev` — Development tools (pytest, black, ruff)
  - `gpu` — CUDA-enabled torch (if needed)

Install optional groups:
```bash
# Install dev dependencies
uv sync --group dev

# Install GPU torch
uv sync --extra gpu
```

## Benefits Over pip

| Feature | uv | pip |
|---------|----|----|
| Dependency Resolution | 10-20x faster | Slow |
| Lock File | ✅ uv.lock (reproducible) | ❌ No lock file |
| Virtual Environment | Auto-created | Manual setup |
| Package Building | ✅ Integrated | ❌ Separate tools |
| Python Version | Auto-managed | Manual |

## Troubleshooting

**Virtual environment not found:**
```bash
uv sync
```

**Dependencies out of sync with uv.lock:**
```bash
uv sync --force
```

**Want to use pip instead (not recommended):**
```bash
pip install -r requirements.txt
```

**Performance issues:**
```bash
# Clear cache if needed
uv cache prune
uv sync --force
```

## File Structure

- `pyproject.toml` — Project metadata and dependency declarations
- `uv.lock` — Locked versions of all dependencies (commit to git)
- `.venv/` — Virtual environment (auto-created, add to .gitignore)
- `requirements.txt` — Alternative pip requirements file (kept for compatibility)

## More Information

- [uv Documentation](https://docs.astral.sh/uv/)
- [PEP 517 (Build System)](https://www.python.org/dev/peps/pep-0517/)
- [PEP 660 (Editable Installs)](https://www.python.org/dev/peps/pep-0660/)
