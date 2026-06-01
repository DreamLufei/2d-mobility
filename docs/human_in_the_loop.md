# Human In The Loop

## Trigger Conditions
- Low-confidence recovery
- Explicit manual-fix recommendation
- Ambiguous refinement decision
- Ambiguous validation decision
- Retry-budget exhaustion

## Runtime Policies
- `interactive`
  - Use terminal prompts when TTY is available.
  - Fall back to response-file waiting if no TTY is present.
  - In tmux, selecting `manual_fix_resume` pauses the run until you type `continue`, `continue <stage>`, `skip`, or `abort`.
  - If no one responds within the review window, the runtime auto-selects `skip_material`.
- `non_interactive_skip_on_timeout`
  - Write payload files and wait for an external response for up to the configured timeout.
  - If no response arrives, resume with `skip_material`.
- `non_interactive_abort_on_timeout`
  - Write payload files and wait for an external response for up to the configured timeout.
  - If no response arrives, resume with `abort_task`.

Legacy aliases:
- `non_interactive_wait` -> `non_interactive_skip_on_timeout`
- `non_interactive_skip` -> `non_interactive_skip_on_timeout`

These aliases remain accepted for compatibility, but runtime normalization records deprecation warnings in environment summaries.

## Payload Files
- `human_escalation_payload.json`
- `human_escalation_response.json`

## Manual-Fix Flow
1. Select `manual_fix_resume` from the escalation menu.
2. Edit files directly in the working directory.
3. Return to the tmux terminal and enter `continue` or `continue <stage>`.
4. Build a typed preview schema containing:
   - `modified_files`
   - `requested_resume_strategy`
   - `computed_resume_stage`
   - `cleanup_policy`
   - `invalidated_stages`
   - `invalidated_artifacts`
   - `warnings`
5. Resume from the chosen stage.

`stage_contracts.py` is the single source of truth for resume validation, cleanup policy interpretation, and invalidation boundaries.
