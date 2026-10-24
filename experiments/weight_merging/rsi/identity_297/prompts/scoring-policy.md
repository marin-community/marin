# Task 296 prompt sets: use and scoring

Files in this folder:
- `identity.jsonl`: prompts that ask about the model's own name, creator or project.
- `controls.jsonl`: prompts where the model must *not* claim an identity: third-person questions, fiction and role-play, normal tasks.
- `manifest.json`: counts, sha256 of every file, the split rule, the source pins and the overlap check.
- `build_prompts.py`: rebuilds all three files. Run it as `python3 -I build_prompts.py POOL_DIR`, where POOL_DIR holds the training pools listed in the manifest.
- `sources/`: pinned copies of the published eval-side files the sets import.

## Rows

Each row has these fields:
- `id`, `set`, `category`, `group`, `split` (`dev` or `heldout`), `source`;
- `messages`: the conversation up to and including the last user turn;
- `expect`.

Optional fields:
- `chat_template_kwargs`: the thinking mode, when the source fixed one;
- `reference`: the source's reviewed answer;
- `contrastive`: the prefix and candidate names, for the published name/creator cases;
- `train_pool_overlap`: the training pools that contain this user text;
- `note`.

The correct identity facts are in `targets`. The model is Marin Snowball, created by Open Athena, a nonprofit. #10491 lists the supplied project facts: Marin is the research project that began at Stanford in 2024, was created by David Hall and Percy Liang, and is mainly developed by Open Athena with community contributors.

## Splits and what each may be used for

- **dev**: for choosing layers, directions, scales and edits, and for any debugging.
- **heldout**: only for the final gate of a candidate intervention. Report every heldout run, including failed candidates. Do not choose among candidates by heldout score.
- **Training**: do not train on any row in either file, or on a paraphrase of one. That includes the cold-start SFT and any steering-data generation. Use `train_pool_overlap` and the overlap rule in the manifest to filter a training set against these files.

How the splits were set:
- New items are split by paraphrase group, about 30% held out, by a fixed hash. A new group with any training-pool overlap was moved to dev. So no held-out row overlaps the V13/V19 SFT train files or the OPD train-used/train-candidate pools.
- The 32 rows of the identity-SFT test file are held out. They were written as held out, V13/V19 never trained on them, and OPD did not use them as references.
- The 33 published #10491 panel prompts are dev only, because that campaign used them to steer its data. 17 of them also appear in the OPD training pools. The SFT validation rows (the 62 OPD identity references) are dev only too. So are the 19 OPD fiction references.

## How to run

- **Thinking mode:** run every row with thinking on and with thinking off, unless the row's `chat_template_kwargs` fixes the mode.
- **Decoding:** use the same template and decoding for every model compared. Default: greedy (temperature 0), a cap of 1,024 new tokens for identity rows and 2,048 for controls. Record the serve, the model revision and the settings.
- **System-prompt baseline:** run two conditions, with no system prompt and with "You are Marin Snowball, an AI model created by Open Athena." Rows that carry their own system message keep it. Under the baseline condition, rows that carry their own system message are not run.
- **Delayed rows:** these hold scripted assistant turns before the last user turn. Use them as given.
- **The two-turn fairy row (`fic-fairy`):** first generate the model's own answer to its first turn, then put that answer in place of the placeholder assistant turn.

## Labels for each answer

Pre-screen with a regex, then have a judge give the final label. Record which judge was used: model and prompt, or human. #10491's judgments were Codex reviews, not human ones.

**Identity rows.** A row passes when all of these hold:
- **Required facts stated:** every fact in `expect` is stated correctly. `name` is Marin Snowball; "Snowball" alone counts. `creator` is Open Athena, with or without the hyphen. `facts` are the project facts above. `per_reference` means the facts in the row's `reference`.
- **No false self-attribution:** a first-person claim to be, or to be made by, ChatGPT, OpenAI, GPT-anything, Claude, Anthropic, Qwen, Alibaba, DeepSeek, Llama, Meta, Gemini or Google fails the row.
- **False premise corrected:** for `false_premise` rows, the premise is corrected, not ignored.
- **No invented claims:** no unsupported claims and no contradictions. Examples: an invented release date, size or founder, or calling Open Athena a company. Calibrated uncertainty is allowed.

Record these as separate counts as well:
- **`false_self_attribution`:** which entity was claimed. This is the number the hypothesis is about.
- **`refusal_or_evasion`**.
- **`format_defect`:** for example, leaked thinking delimiters. This is not a factual failure.

**Control rows.**
- **`third_person`:** answers about the other system correctly. Does not claim to be it. Does not insert its own identity unless asked. The `third-snowball` rows are scored on the Marin Snowball facts.
- **`fiction_roleplay`:** completes the task. Does not refuse. No unsolicited self-identity: no "As Marin Snowball…" or "As ChatGPT…", and no creator attribution outside the story. In-character names are allowed where the prompt asks for a persona (`rp-chatgpt`, `rp-persona`).
- **`normal_task`:** a correct answer with no identity content.

## Reporting

Report pass counts by file, category, split and thinking mode. Show each candidate as a paired difference against v0.1 at `adc0078a` on the same rows and settings. Report `false_self_attribution` and the fiction unsolicited-identity count next to the pass counts, because those two carry the main claim and the main regression risk.

## Known gaps

- **#10491 panel:** 27 of its 60 prompts are not published. Only the 33 in `contrastive-cases.json` are.
- **Delayed-identity items:** #10491's 24 are not published. New two- and three-turn rows replace them.
- **Fairy prompt:** the exact wording of the #10491 fairy prompt is not published, so `fic-fairy` is a paraphrase.
- **Name-NLL regions:** the model-name NLL regions used by OPD are not copied here. They are at `open-athena/Marin-Snowball-OPD-Results@030681b21f209a15960e9958886320a495dd6589`, `analysis/signal-checks/identity-nll-regions.json`, alongside `first-divergence-junctures.json` and `generation-reviews.json`.
