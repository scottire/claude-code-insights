Analyze this Claude Code session and extract structured facets.

CRITICAL GUIDELINES:

1. **goal_categories**: Count ONLY what the USER explicitly asked for. Use ONLY these categories:
   - debug_investigate: Debugging or investigating an issue
   - implement_feature: Building new functionality
   - fix_bug: Fixing a bug
   - write_script_tool: Writing a script or tool
   - refactor_code: Refactoring existing code
   - configure_system: System/environment configuration
   - create_pr_commit: Creating commits or PRs
   - analyze_data: Data analysis
   - understand_codebase: Understanding code
   - write_tests: Writing tests
   - write_docs: Writing documentation
   - deploy_infra: Deployment/infrastructure
   - warmup_minimal: Very short/warmup session

2. **user_satisfaction_counts**: Base ONLY on explicit user signals.
   - happy: "Yay!", "great!", "perfect!"
   - satisfied: "thanks", "looks good", "that works"
   - likely_satisfied: "ok, now let's..." (continuing without complaint)
   - dissatisfied: "that's not right", "try again"
   - frustrated: "this is broken", "I give up"

3. **friction_counts**: Use ONLY these categories:
   - misunderstood_request: Claude interpreted incorrectly
   - wrong_approach: Right goal, wrong solution method
   - buggy_code: Code didn't work correctly
   - user_rejected_action: User said no/stop to a tool call
   - claude_got_blocked: Claude couldn't proceed
   - user_stopped_early: User ended session early
   - wrong_file_or_location: Wrong file or location
   - excessive_changes: Over-engineered or changed too much
   - slow_or_verbose: Too slow or verbose
   - tool_failed: A tool call failed
   - user_unclear: User's request was unclear
   - external_issue: External system issue

4. **primary_success**: Use ONLY these categories:
   - none: No notable success
   - fast_accurate_search: Quick, accurate code search
   - correct_code_edits: Correct code edits
   - good_explanations: Clear explanations
   - proactive_help: Proactive assistance
   - multi_file_changes: Good multi-file changes
   - good_debugging: Effective debugging

5. **user_instructions_to_claude**: Extract explicit instructions the user gave.
   - Look for "always...", "never...", "make sure to...", "don't forget..."
   - Examples: "always run tests", "use TypeScript", "keep it simple"
   - Return empty array if none found

SESSION:
