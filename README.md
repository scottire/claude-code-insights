# Claude Code Insights

Analyze your Claude Code session transcripts and generate insight reports about usage patterns, friction points, and suggestions for improvement.

## Features

- **Facet Extraction** - Analyzes each session to extract goals, outcomes, friction, and satisfaction
- **Aggregation** - Combines facets into statistics
- **Analysis** - Runs 7 parallel LLM prompts for deep analysis (project areas, interaction style, what works, friction, suggestions, horizon, fun ending)
- **Synthesis** - Generates an "At a Glance" summary
- **HTML Report** - Beautiful, interactive report with charts and copy-to-clipboard suggestions

## Quick Start (Marimo Notebook)

Run the interactive notebook in sandbox mode:

```bash
uvx marimo edit --sandbox https://raw.githubusercontent.com/scottire/claude-code-insights/main/insights_sandbox.py
```

## Installation

```bash
pip install git+https://github.com/scottire/claude-code-insights.git
```

## CLI Usage

```bash
# Analyze your local Claude Code sessions
insights run-local --limit 10

# Or specify a custom path
insights run-all -t ~/.claude/projects -o output/report.json
```

## Python API

```python
from insights import init_weave, run_insights_pipeline, load_transcripts
from pathlib import Path

# Initialize Weave tracing (optional but recommended)
init_weave("my-project")

# Load transcripts
transcripts, date_range = load_transcripts(
    Path.home() / ".claude" / "projects",
    limit=10
)

# Run pipeline
html_bytes = run_insights_pipeline(
    transcripts,
    api_key="your-wandb-api-key",  # or set WANDB_API_KEY env var
    workers=5
)

# Save report
Path("report.html").write_bytes(html_bytes)
```

## Requirements

- Python 3.11+
- W&B API key (get from [wandb.ai/settings](https://wandb.ai/settings))

## Viewing Traces

All LLM calls are traced via [Weave](https://wandb.ai/site/weave). After running:

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to your project (default: `claude-code-insights`)
3. Click "Weave" tab to see all traced calls

## License

MIT
