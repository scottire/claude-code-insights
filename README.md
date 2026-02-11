# Claude Code Insights

Reverse-engineered `/insights` pipeline from Claude Code. Analyzes your session transcripts and generates a report about how you work.

## Run it

```bash
uvx marimo edit --sandbox 'https://github.com/scottire/claude-code-insights/blob/main/notebook.py'
```

You'll need a [W&B API key](https://wandb.ai/authorize).

## What it does

1. **Extract facets** - LLM reads each session, outputs structured JSON (goals, outcomes, friction)
2. **Aggregate** - Pure Python, merges counts across sessions
3. **Analyze** - 7 prompts run in parallel (project areas, what works, friction, suggestions, etc.)
4. **Synthesize** - Condenses into a summary report

The notebook lets you step through each stage or just run the whole pipeline.

## Traces

All LLM calls are traced via [Weave](https://wandb.ai/site/weave). The final report renders as a view on the pipeline call.
