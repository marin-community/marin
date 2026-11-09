

# This session

You work in a workspace through tools; you do not reply with the program. The workspace holds:
- `{program}`: a copy of the template module. Adapt it to the proposal; it is what you submit.
- `{proposal}`: the proposal. `{sdk}` and `{template}`: the SDK reference and the template (read-only; `read_reference` reads them too).
- `{notes}`: your notes. If this session runs out of turns, a fresh session starts from this workspace and reads them.

The loop: adapt `{program}` with `edit_file`, check it with `run_loader`, run it with `run_build`, read the failure, fix the step it names, and run it again. The step store reruns only the steps you changed and the steps after them, so a rerun costs the edited steps alone. Once it builds, grade a wrong answer or two against the built task with `run_grade_candidate`, then `submit` the program. `{sdk}` lists every name, keyword argument and field the program may use: read it instead of probing, since a step that raises to print module names, fields or schemas spends a run and shows less than the reference. Never rewrite the whole module in one call; edit the step you are fixing (`write_file` takes at most {write_chars} characters). The session ends only with `submit`: a reply without a tool call hands nothing in. You have {turns} turns.
You write the solver instruction yourself as a string in the `instructions` step: it follows the instruction rule above (format as requirements, never grader behaviour), and `run_build` reports one that names the grading on its `instruction:` line; rewrite it before you submit.

