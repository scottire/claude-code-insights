"""
HTML Report Template for Claude Code Insights.

This module contains the CSS, JavaScript, and HTML generation functions
that replicate the original cli.js report format.
"""

import re
from typing import Any

# ============================================================================
# Label Mapping (from cli.js)
# ============================================================================

LABEL_MAP = {
    "debug_investigate": "Debug/Investigate",
    "implement_feature": "Implement Feature",
    "fix_bug": "Fix Bug",
    "write_script_tool": "Write Script/Tool",
    "refactor_code": "Refactor Code",
    "configure_system": "Configure System",
    "create_pr_commit": "Create PR/Commit",
    "analyze_data": "Analyze Data",
    "understand_codebase": "Understand Codebase",
    "write_tests": "Write Tests",
    "write_docs": "Write Docs",
    "deploy_infra": "Deploy/Infra",
    "warmup_minimal": "Cache Warmup",
    "fast_accurate_search": "Fast/Accurate Search",
    "correct_code_edits": "Correct Code Edits",
    "good_explanations": "Good Explanations",
    "proactive_help": "Proactive Help",
    "multi_file_changes": "Multi-file Changes",
    "handled_complexity": "Multi-file Changes",
    "good_debugging": "Good Debugging",
    "misunderstood_request": "Misunderstood Request",
    "wrong_approach": "Wrong Approach",
    "buggy_code": "Buggy Code",
    "user_rejected_action": "User Rejected Action",
    "claude_got_blocked": "Claude Got Blocked",
    "user_stopped_early": "User Stopped Early",
    "wrong_file_or_location": "Wrong File/Location",
    "excessive_changes": "Excessive Changes",
    "slow_or_verbose": "Slow/Verbose",
    "tool_failed": "Tool Failed",
    "user_unclear": "User Unclear",
    "external_issue": "External Issue",
    "frustrated": "Frustrated",
    "dissatisfied": "Dissatisfied",
    "likely_satisfied": "Likely Satisfied",
    "satisfied": "Satisfied",
    "happy": "Happy",
    "unsure": "Unsure",
    "neutral": "Neutral",
    "delighted": "Delighted",
    "single_task": "Single Task",
    "multi_task": "Multi Task",
    "iterative_refinement": "Iterative Refinement",
    "exploration": "Exploration",
    "quick_question": "Quick Question",
    "fully_achieved": "Fully Achieved",
    "mostly_achieved": "Mostly Achieved",
    "partially_achieved": "Partially Achieved",
    "not_achieved": "Not Achieved",
    "unclear_from_transcript": "Unclear",
    "unhelpful": "Unhelpful",
    "slightly_helpful": "Slightly Helpful",
    "moderately_helpful": "Moderately Helpful",
    "very_helpful": "Very Helpful",
    "essential": "Essential",
}

# ============================================================================
# CSS Styles (exact copy from cli.js)
# ============================================================================

CSS = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #f8fafc; color: #334155; line-height: 1.65; padding: 48px 24px; }
    .container { max-width: 800px; margin: 0 auto; }
    h1 { font-size: 32px; font-weight: 700; color: #0f172a; margin-bottom: 8px; }
    h2 { font-size: 20px; font-weight: 600; color: #0f172a; margin-top: 48px; margin-bottom: 16px; }
    .subtitle { color: #64748b; font-size: 15px; margin-bottom: 32px; }
    .nav-toc { display: flex; flex-wrap: wrap; gap: 8px; margin: 24px 0 32px 0; padding: 16px; background: white; border-radius: 8px; border: 1px solid #e2e8f0; }
    .nav-toc a { font-size: 12px; color: #64748b; text-decoration: none; padding: 6px 12px; border-radius: 6px; background: #f1f5f9; transition: all 0.15s; }
    .nav-toc a:hover { background: #e2e8f0; color: #334155; }
    .stats-row { display: flex; gap: 24px; margin-bottom: 40px; padding: 20px 0; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; flex-wrap: wrap; }
    .stat { text-align: center; }
    .stat-value { font-size: 24px; font-weight: 700; color: #0f172a; }
    .stat-label { font-size: 11px; color: #64748b; text-transform: uppercase; }
    .at-a-glance { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 1px solid #f59e0b; border-radius: 12px; padding: 20px 24px; margin-bottom: 32px; }
    .glance-title { font-size: 16px; font-weight: 700; color: #92400e; margin-bottom: 16px; }
    .glance-sections { display: flex; flex-direction: column; gap: 12px; }
    .glance-section { font-size: 14px; color: #78350f; line-height: 1.6; }
    .glance-section strong { color: #92400e; }
    .see-more { color: #b45309; text-decoration: none; font-size: 13px; white-space: nowrap; }
    .see-more:hover { text-decoration: underline; }
    .project-areas { display: flex; flex-direction: column; gap: 12px; margin-bottom: 32px; }
    .project-area { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
    .area-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .area-name { font-weight: 600; font-size: 15px; color: #0f172a; }
    .area-count { font-size: 12px; color: #64748b; background: #f1f5f9; padding: 2px 8px; border-radius: 4px; }
    .area-desc { font-size: 14px; color: #475569; line-height: 1.5; }
    .narrative { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 24px; }
    .narrative p { margin-bottom: 12px; font-size: 14px; color: #475569; line-height: 1.7; }
    .key-insight { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 14px; color: #166534; }
    .section-intro { font-size: 14px; color: #64748b; margin-bottom: 16px; }
    .big-wins { display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px; }
    .big-win { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 16px; }
    .big-win-title { font-weight: 600; font-size: 15px; color: #166534; margin-bottom: 8px; }
    .big-win-desc { font-size: 14px; color: #15803d; line-height: 1.5; }
    .friction-categories { display: flex; flex-direction: column; gap: 16px; margin-bottom: 24px; }
    .friction-category { background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 16px; }
    .friction-title { font-weight: 600; font-size: 15px; color: #991b1b; margin-bottom: 6px; }
    .friction-desc { font-size: 13px; color: #7f1d1d; margin-bottom: 10px; }
    .friction-examples { margin: 0 0 0 20px; font-size: 13px; color: #334155; }
    .friction-examples li { margin-bottom: 4px; }
    .claude-md-section { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
    .claude-md-section h3 { font-size: 14px; font-weight: 600; color: #1e40af; margin: 0 0 12px 0; }
    .claude-md-actions { margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #dbeafe; }
    .copy-all-btn { background: #2563eb; color: white; border: none; border-radius: 4px; padding: 6px 12px; font-size: 12px; cursor: pointer; font-weight: 500; transition: all 0.2s; }
    .copy-all-btn:hover { background: #1d4ed8; }
    .copy-all-btn.copied { background: #16a34a; }
    .claude-md-item { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px; padding: 10px 0; border-bottom: 1px solid #dbeafe; }
    .claude-md-item:last-child { border-bottom: none; }
    .cmd-checkbox { margin-top: 2px; }
    .cmd-code { background: white; padding: 8px 12px; border-radius: 4px; font-size: 12px; color: #1e40af; border: 1px solid #bfdbfe; font-family: monospace; display: block; white-space: pre-wrap; word-break: break-word; flex: 1; }
    .cmd-why { font-size: 12px; color: #64748b; width: 100%; padding-left: 24px; margin-top: 4px; }
    .features-section, .patterns-section { display: flex; flex-direction: column; gap: 12px; margin: 16px 0; }
    .feature-card { background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 16px; }
    .pattern-card { background: #f0f9ff; border: 1px solid #7dd3fc; border-radius: 8px; padding: 16px; }
    .feature-title, .pattern-title { font-weight: 600; font-size: 15px; color: #0f172a; margin-bottom: 6px; }
    .feature-oneliner { font-size: 14px; color: #475569; margin-bottom: 8px; }
    .pattern-summary { font-size: 14px; color: #475569; margin-bottom: 8px; }
    .feature-why, .pattern-detail { font-size: 13px; color: #334155; line-height: 1.5; }
    .feature-examples { margin-top: 12px; }
    .feature-example { padding: 8px 0; border-top: 1px solid #d1fae5; }
    .feature-example:first-child { border-top: none; }
    .example-desc { font-size: 13px; color: #334155; margin-bottom: 6px; }
    .example-code-row { display: flex; align-items: flex-start; gap: 8px; }
    .example-code { flex: 1; background: #f1f5f9; padding: 8px 12px; border-radius: 4px; font-family: monospace; font-size: 12px; color: #334155; overflow-x: auto; white-space: pre-wrap; }
    .copyable-prompt-section { margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0; }
    .copyable-prompt-row { display: flex; align-items: flex-start; gap: 8px; }
    .copyable-prompt { flex: 1; background: #f8fafc; padding: 10px 12px; border-radius: 4px; font-family: monospace; font-size: 12px; color: #334155; border: 1px solid #e2e8f0; white-space: pre-wrap; line-height: 1.5; }
    .feature-code { background: #f8fafc; padding: 12px; border-radius: 6px; margin-top: 12px; border: 1px solid #e2e8f0; display: flex; align-items: flex-start; gap: 8px; }
    .feature-code code { flex: 1; font-family: monospace; font-size: 12px; color: #334155; white-space: pre-wrap; }
    .pattern-prompt { background: #f8fafc; padding: 12px; border-radius: 6px; margin-top: 12px; border: 1px solid #e2e8f0; }
    .pattern-prompt code { font-family: monospace; font-size: 12px; color: #334155; display: block; white-space: pre-wrap; margin-bottom: 8px; }
    .prompt-label { font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }
    .copy-btn { background: #e2e8f0; border: none; border-radius: 4px; padding: 4px 8px; font-size: 11px; cursor: pointer; color: #475569; flex-shrink: 0; }
    .copy-btn:hover { background: #cbd5e1; }
    .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }
    .chart-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
    .chart-title { font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 12px; }
    .bar-row { display: flex; align-items: center; margin-bottom: 6px; }
    .bar-label { width: 100px; font-size: 11px; color: #475569; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bar-track { flex: 1; height: 6px; background: #f1f5f9; border-radius: 3px; margin: 0 8px; }
    .bar-fill { height: 100%; border-radius: 3px; }
    .bar-value { width: 28px; font-size: 11px; font-weight: 500; color: #64748b; text-align: right; }
    .empty { color: #94a3b8; font-size: 13px; }
    .horizon-section { display: flex; flex-direction: column; gap: 16px; }
    .horizon-card { background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%); border: 1px solid #c4b5fd; border-radius: 8px; padding: 16px; }
    .horizon-title { font-weight: 600; font-size: 15px; color: #5b21b6; margin-bottom: 8px; }
    .horizon-possible { font-size: 14px; color: #334155; margin-bottom: 10px; line-height: 1.5; }
    .horizon-tip { font-size: 13px; color: #6b21a8; background: rgba(255,255,255,0.6); padding: 8px 12px; border-radius: 4px; }
    .fun-ending { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 1px solid #fbbf24; border-radius: 12px; padding: 24px; margin-top: 40px; text-align: center; }
    .fun-headline { font-size: 18px; font-weight: 600; color: #78350f; margin-bottom: 8px; }
    .fun-detail { font-size: 14px; color: #92400e; }
    .collapsible-section { margin-top: 16px; }
    .collapsible-header { display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 12px 0; border-bottom: 1px solid #e2e8f0; }
    .collapsible-header h3 { margin: 0; font-size: 14px; font-weight: 600; color: #475569; }
    .collapsible-arrow { font-size: 12px; color: #94a3b8; transition: transform 0.2s; }
    .collapsible-content { display: none; padding-top: 16px; }
    .collapsible-content.open { display: block; }
    .collapsible-header.open .collapsible-arrow { transform: rotate(90deg); }
    @media (max-width: 640px) { .charts-row { grid-template-columns: 1fr; } .stats-row { justify-content: center; } }
"""

# ============================================================================
# JavaScript (for interactivity)
# ============================================================================

JAVASCRIPT = """
    function toggleCollapsible(header) {
      header.classList.toggle('open');
      const content = header.nextElementSibling;
      content.classList.toggle('open');
    }
    function copyText(btn) {
      const code = btn.previousElementSibling;
      navigator.clipboard.writeText(code.textContent).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
      });
    }
    function copyCmdItem(idx) {
      const checkbox = document.getElementById('cmd-' + idx);
      if (checkbox) {
        const text = checkbox.dataset.text;
        navigator.clipboard.writeText(text).then(() => {
          const btn = checkbox.nextElementSibling.querySelector('.copy-btn');
          if (btn) { btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = 'Copy'; }, 2000); }
        });
      }
    }
    function copyAllCheckedClaudeMd() {
      const checkboxes = document.querySelectorAll('.cmd-checkbox:checked');
      const texts = [];
      checkboxes.forEach(cb => {
        if (cb.dataset.text) { texts.push(cb.dataset.text); }
      });
      const combined = texts.join('\\n');
      const btn = document.querySelector('.copy-all-btn');
      if (btn) {
        navigator.clipboard.writeText(combined).then(() => {
          btn.textContent = 'Copied ' + texts.length + ' items!';
          btn.classList.add('copied');
          setTimeout(() => { btn.textContent = 'Copy All Checked'; btn.classList.remove('copied'); }, 2000);
        });
      }
    }
"""


# ============================================================================
# Helper Functions
# ============================================================================

def html_escape(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;"))


def html_escape_with_bold(text: str) -> str:
    """Escape HTML but preserve **bold** markers."""
    if not text:
        return ""
    escaped = html_escape(text)
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)


def get_label(key: str) -> str:
    """Get human-readable label for a key."""
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    return key.replace("_", " ").replace("-", " ").title()


def generate_bar_chart(data: dict, color: str, max_items: int = 6, order: list = None) -> str:
    """Generate bar chart HTML from data dict."""
    if order:
        items = [(k, data.get(k, 0)) for k in order if k in data and data.get(k, 0) > 0]
    else:
        items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:max_items]

    if not items:
        return '<p class="empty">No data</p>'

    max_val = max(v for _, v in items) if items else 1

    rows = []
    for key, value in items:
        pct = (value / max_val) * 100
        label = get_label(key)
        rows.append(f'''<div class="bar-row">
        <div class="bar-label">{html_escape(label)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>
        <div class="bar-value">{value}</div>
      </div>''')

    return "\n".join(rows)


# ============================================================================
# Section Generators
# ============================================================================

def get_text_value(value) -> str:
    """Extract text from a value that might be a string or nested dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Handle nested dicts like {"whats_working": "text"}
        for v in value.values():
            if isinstance(v, str):
                return v
        return str(value)
    return str(value) if value else ""


def generate_at_a_glance_section(at_a_glance: dict) -> str:
    """Generate the at-a-glance section HTML."""
    if not at_a_glance:
        return ""

    sections = []
    if at_a_glance.get("whats_working"):
        text = get_text_value(at_a_glance["whats_working"])
        sections.append(f'<div class="glance-section"><strong>What\'s working:</strong> {html_escape_with_bold(text)} <a href="#section-wins" class="see-more">Impressive Things You Did →</a></div>')
    if at_a_glance.get("whats_hindering"):
        text = get_text_value(at_a_glance["whats_hindering"])
        sections.append(f'<div class="glance-section"><strong>What\'s hindering you:</strong> {html_escape_with_bold(text)} <a href="#section-friction" class="see-more">Where Things Go Wrong →</a></div>')
    if at_a_glance.get("quick_wins"):
        text = get_text_value(at_a_glance["quick_wins"])
        sections.append(f'<div class="glance-section"><strong>Quick wins to try:</strong> {html_escape_with_bold(text)} <a href="#section-features" class="see-more">Features to Try →</a></div>')
    if at_a_glance.get("ambitious_workflows"):
        text = get_text_value(at_a_glance["ambitious_workflows"])
        sections.append(f'<div class="glance-section"><strong>Ambitious workflows:</strong> {html_escape_with_bold(text)} <a href="#section-horizon" class="see-more">On the Horizon →</a></div>')

    if not sections:
        return ""

    return f'''
    <div class="at-a-glance">
      <div class="glance-title">At a Glance</div>
      <div class="glance-sections">
        {"".join(sections)}
      </div>
    </div>
    '''


def generate_project_areas_section(analysis: dict) -> str:
    """Generate the project areas section HTML."""
    project_areas = analysis.get("project_areas", {})
    areas = project_areas.get("areas", []) if isinstance(project_areas.get("areas"), list) else []

    if not areas:
        return ""

    area_items = []
    for area in areas:
        if isinstance(area, dict):
            area_items.append(f'''
        <div class="project-area">
          <div class="area-header">
            <span class="area-name">{html_escape(area.get("name", ""))}</span>
            <span class="area-count">~{area.get("session_count", 0)} sessions</span>
          </div>
          <div class="area-desc">{html_escape(area.get("description", ""))}</div>
        </div>''')

    return f'''
    <h2 id="section-work">What You Work On</h2>
    <div class="project-areas">
      {"".join(area_items)}
    </div>
    '''


def generate_interaction_style_section(analysis: dict) -> str:
    """Generate the interaction style section HTML."""
    interaction = analysis.get("interaction_style", {})

    if not interaction.get("narrative"):
        return ""

    # Convert paragraphs
    paragraphs = interaction["narrative"].split("\n\n")
    para_html = ""
    for p in paragraphs:
        escaped = html_escape(p)
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
        escaped = escaped.replace("- ", "• ")
        escaped = escaped.replace("\n", "<br>")
        para_html += f"<p>{escaped}</p>\n"

    key_insight = ""
    if interaction.get("key_pattern"):
        key_insight = f'<div class="key-insight"><strong>Key pattern:</strong> {html_escape(interaction["key_pattern"])}</div>'

    return f'''
    <h2 id="section-usage">How You Use Claude Code</h2>
    <div class="narrative">
      {para_html}
      {key_insight}
    </div>
    '''


def generate_what_works_section(analysis: dict) -> str:
    """Generate the what works / impressive things section HTML."""
    what_works = analysis.get("what_works", {})
    workflows = what_works.get("impressive_workflows", [])

    if not workflows:
        return ""

    intro = f'<p class="section-intro">{html_escape(what_works.get("intro", ""))}</p>' if what_works.get("intro") else ""
    wins = []
    for wf in workflows:
        if isinstance(wf, dict):
            wins.append(f'''
        <div class="big-win">
          <div class="big-win-title">{html_escape(wf.get("title", ""))}</div>
          <div class="big-win-desc">{html_escape(wf.get("description", ""))}</div>
        </div>''')

    return f'''
    <h2 id="section-wins">Impressive Things You Did</h2>
    {intro}
    <div class="big-wins">
      {"".join(wins)}
    </div>
    '''


def generate_friction_section(analysis: dict) -> str:
    """Generate the friction analysis section HTML."""
    friction = analysis.get("friction_analysis", {})
    categories = friction.get("categories", [])

    if not categories:
        return ""

    intro = f'<p class="section-intro">{html_escape(friction.get("intro", ""))}</p>' if friction.get("intro") else ""
    cats = []
    for cat in categories:
        if isinstance(cat, dict):
            examples = cat.get("examples", [])
            examples_html = "".join([f"<li>{html_escape(ex)}</li>" for ex in examples]) if examples else ""
            cats.append(f'''
        <div class="friction-category">
          <div class="friction-title">{html_escape(cat.get("category", ""))}</div>
          <div class="friction-desc">{html_escape(cat.get("description", ""))}</div>
          {f'<ul class="friction-examples">{examples_html}</ul>' if examples_html else ""}
        </div>''')

    return f'''
    <h2 id="section-friction">Where Things Go Wrong</h2>
    {intro}
    <div class="friction-categories">
      {"".join(cats)}
    </div>
    '''


def generate_suggestions_section(analysis: dict) -> str:
    """Generate the suggestions section HTML."""
    suggestions = analysis.get("suggestions", {})

    # Claude.md additions
    claude_md_additions = suggestions.get("claude_md_additions", [])
    claude_md_html = ""
    if claude_md_additions:
        items = []
        for i, item in enumerate(claude_md_additions):
            if isinstance(item, dict):
                data_text = html_escape(f"{item.get('prompt_scaffold', item.get('where', 'Add to CLAUDE.md'))}\n\n{item.get('addition', '')}")
                items.append(f'''
        <div class="claude-md-item">
          <input type="checkbox" id="cmd-{i}" class="cmd-checkbox" checked data-text="{data_text}">
          <label for="cmd-{i}">
            <code class="cmd-code">{html_escape(item.get("addition", ""))}</code>
            <button class="copy-btn" onclick="copyCmdItem({i})">Copy</button>
          </label>
          <div class="cmd-why">{html_escape(item.get("why", ""))}</div>
        </div>''')
        claude_md_html = f'''
    <div class="claude-md-section">
      <h3>Suggested CLAUDE.md Additions</h3>
      <p style="font-size: 12px; color: #64748b; margin-bottom: 12px;">Just copy this into Claude Code to add it to your CLAUDE.md.</p>
      <div class="claude-md-actions">
        <button class="copy-all-btn" onclick="copyAllCheckedClaudeMd()">Copy All Checked</button>
      </div>
      {"".join(items)}
    </div>
    '''

    # Features to try
    features = suggestions.get("features_to_try", [])
    features_html = ""
    if features:
        cards = []
        for f in features:
            if isinstance(f, dict):
                code_section = ""
                if f.get("example_code"):
                    code_section = f'''
          <div class="feature-examples">
            <div class="feature-example">
              <div class="example-code-row">
                <code class="example-code">{html_escape(f["example_code"])}</code>
                <button class="copy-btn" onclick="copyText(this)">Copy</button>
              </div>
            </div>
          </div>'''
                cards.append(f'''
        <div class="feature-card">
          <div class="feature-title">{html_escape(f.get("feature", ""))}</div>
          <div class="feature-oneliner">{html_escape(f.get("one_liner", ""))}</div>
          <div class="feature-why"><strong>Why for you:</strong> {html_escape(f.get("why_for_you", ""))}</div>
          {code_section}
        </div>''')
        features_html = f'''
    <p style="font-size: 13px; color: #64748b; margin-bottom: 12px;">Just copy this into Claude Code and it'll set it up for you.</p>
    <div class="features-section">
      {"".join(cards)}
    </div>
    '''

    if not claude_md_html and not features_html:
        return ""

    return f'''
    <h2 id="section-features">Existing CC Features to Try</h2>
    {claude_md_html}
    {features_html}
    '''


def generate_patterns_section(analysis: dict) -> str:
    """Generate the usage patterns section HTML."""
    suggestions = analysis.get("suggestions", {})
    patterns = suggestions.get("usage_patterns", [])

    if not patterns:
        return ""

    cards = []
    for p in patterns:
        if isinstance(p, dict):
            prompt_section = ""
            if p.get("copyable_prompt"):
                prompt_section = f'''
          <div class="copyable-prompt-section">
            <div class="prompt-label">Paste into Claude Code:</div>
            <div class="copyable-prompt-row">
              <code class="copyable-prompt">{html_escape(p["copyable_prompt"])}</code>
              <button class="copy-btn" onclick="copyText(this)">Copy</button>
            </div>
          </div>'''
            detail = f'<div class="pattern-detail">{html_escape(p.get("detail", ""))}</div>' if p.get("detail") else ""
            cards.append(f'''
        <div class="pattern-card">
          <div class="pattern-title">{html_escape(p.get("title", ""))}</div>
          <div class="pattern-summary">{html_escape(p.get("suggestion", ""))}</div>
          {detail}
          {prompt_section}
        </div>''')

    return f'''
    <h2 id="section-patterns">New Ways to Use Claude Code</h2>
    <p style="font-size: 13px; color: #64748b; margin-bottom: 12px;">Just copy this into Claude Code and it'll walk you through it.</p>
    <div class="patterns-section">
      {"".join(cards)}
    </div>
    '''


def generate_horizon_section(analysis: dict) -> str:
    """Generate the on the horizon section HTML."""
    horizon = analysis.get("on_the_horizon", {})
    opportunities = horizon.get("opportunities", [])

    if not opportunities:
        return ""

    intro = f'<p class="section-intro">{html_escape(horizon.get("intro", ""))}</p>' if horizon.get("intro") else ""
    cards = []
    for opp in opportunities:
        if isinstance(opp, dict):
            tip = f'<div class="horizon-tip"><strong>Getting started:</strong> {html_escape(opp.get("how_to_try", ""))}</div>' if opp.get("how_to_try") else ""
            prompt_section = ""
            if opp.get("copyable_prompt"):
                prompt_section = f'''<div class="pattern-prompt"><div class="prompt-label">Paste into Claude Code:</div><code>{html_escape(opp["copyable_prompt"])}</code><button class="copy-btn" onclick="copyText(this)">Copy</button></div>'''
            cards.append(f'''
        <div class="horizon-card">
          <div class="horizon-title">{html_escape(opp.get("title", ""))}</div>
          <div class="horizon-possible">{html_escape(opp.get("whats_possible", ""))}</div>
          {tip}
          {prompt_section}
        </div>''')

    return f'''
    <h2 id="section-horizon">On the Horizon</h2>
    {intro}
    <div class="horizon-section">
      {"".join(cards)}
    </div>
    '''


def generate_fun_ending_section(analysis: dict) -> str:
    """Generate the fun ending section HTML."""
    fun = analysis.get("fun_ending", {})

    if not fun.get("headline"):
        return ""

    detail = f'<div class="fun-detail">{html_escape(fun.get("detail", ""))}</div>' if fun.get("detail") else ""

    return f'''
    <div class="fun-ending">
      <div class="fun-headline">"{html_escape(fun["headline"])}"</div>
      {detail}
    </div>
    '''


# ============================================================================
# Main Report Generator
# ============================================================================

def generate_html_report(report: dict) -> str:
    """Generate HTML report matching the original cli.js format."""
    at_a_glance = report.get("at_a_glance", {})
    analysis = report.get("analysis", {})
    aggregated = report.get("aggregated", {})

    # Generate sections
    at_a_glance_html = generate_at_a_glance_section(at_a_glance)
    project_areas_html = generate_project_areas_section(analysis)
    interaction_html = generate_interaction_style_section(analysis)
    what_works_html = generate_what_works_section(analysis)
    friction_html = generate_friction_section(analysis)
    suggestions_html = generate_suggestions_section(analysis)
    patterns_html = generate_patterns_section(analysis)
    horizon_html = generate_horizon_section(analysis)
    fun_html = generate_fun_ending_section(analysis)

    # Chart data
    goal_categories = aggregated.get("goal_categories", {})
    session_types = aggregated.get("session_types", {})
    friction_counts = aggregated.get("friction", {})
    outcomes = aggregated.get("outcomes", {})
    satisfaction = aggregated.get("satisfaction", {})

    # Metadata
    total_sessions = aggregated.get("total_sessions", 0) or report.get("session_count", 0)
    # Use report's date_range string if available, otherwise fall back to aggregated
    date_range_str = report.get("date_range", "")
    if date_range_str and date_range_str != "unknown":
        start_date, end_date = date_range_str.split(" to ") if " to " in date_range_str else ("", "")
    else:
        date_range = aggregated.get("date_range", {})
        start_date = date_range.get("start", "")
        end_date = date_range.get("end", "")

    # Build final HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Claude Code Insights</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
{CSS}
  </style>
</head>
<body>
  <div class="container">
    <h1>Claude Code Insights</h1>
    <p class="subtitle">{total_sessions} sessions | {start_date or "?"} to {end_date or "?"}</p>

    {at_a_glance_html}

    <nav class="nav-toc">
      <a href="#section-work">What You Work On</a>
      <a href="#section-usage">How You Use CC</a>
      <a href="#section-wins">Impressive Things</a>
      <a href="#section-friction">Where Things Go Wrong</a>
      <a href="#section-features">Features to Try</a>
      <a href="#section-patterns">New Usage Patterns</a>
      <a href="#section-horizon">On the Horizon</a>
    </nav>

    <div class="stats-row">
      <div class="stat"><div class="stat-value">{total_sessions}</div><div class="stat-label">Sessions</div></div>
      <div class="stat"><div class="stat-value">{aggregated.get("days_active", 0)}</div><div class="stat-label">Days</div></div>
    </div>

    {project_areas_html}

    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">What You Wanted</div>
        {generate_bar_chart(goal_categories, "#2563eb")}
      </div>
      <div class="chart-card">
        <div class="chart-title">Session Types</div>
        {generate_bar_chart(session_types, "#8b5cf6")}
      </div>
    </div>

    {interaction_html}

    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">What Helped Most</div>
        {generate_bar_chart(aggregated.get("primary_successes", {}), "#16a34a")}
      </div>
      <div class="chart-card">
        <div class="chart-title">Outcomes</div>
        {generate_bar_chart(outcomes, "#8b5cf6", 6, ["fully_achieved", "mostly_achieved", "partially_achieved", "not_achieved", "unclear_from_transcript"])}
      </div>
    </div>

    {what_works_html}

    {friction_html}

    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">Primary Friction Types</div>
        {generate_bar_chart(friction_counts, "#dc2626")}
      </div>
      <div class="chart-card">
        <div class="chart-title">Inferred Satisfaction</div>
        {generate_bar_chart(satisfaction, "#eab308", 6, ["frustrated", "dissatisfied", "likely_satisfied", "satisfied", "happy", "unsure"])}
      </div>
    </div>

    {suggestions_html}

    {patterns_html}

    {horizon_html}

    {fun_html}
  </div>
  <script>{JAVASCRIPT}</script>
</body>
</html>'''

    return html
