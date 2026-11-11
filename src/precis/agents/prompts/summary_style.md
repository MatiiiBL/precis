## What a good summary looks like

Write for a technical reader in plain, precise English. Say what was done and what was measured, not how important it is.

- **Specific and quantitative.** Prefer "reduces L2 miss rate by 18% over LRU on SPEC CPU2017" to "significantly improves cache performance". Name the baseline, the metric, the workload and the setting each number was measured in.
- **Numbers come only from the paper.** Copy figures exactly as the paper states them, with their units and precision. Never estimate, round, convert or compute a number the paper does not state, and never fill a gap with a plausible value. If the paper gives no number for a result, describe the result without one.
- **Scope exactly as the paper does.** If a result holds for one dataset, model size or configuration, say so; do not generalize it. Keep what was shown experimentally apart from what the authors argue or expect.
- **No promotional language.** Avoid words such as novel, groundbreaking, state-of-the-art, significant (unless it is statistical), remarkable, powerful or seamless, and do not repeat the authors' adjectives about their own work.
- **Self-contained.** Expand an acronym the first time it appears unless it is universal (GPU, CPU, LLM). Name the method instead of writing "the proposed method".

The fields:

- `tldr`: one sentence of at most 40 words: what the paper does and its headline result, with the number when there is one.
- `problem`: the problem and why it matters, in 1-3 sentences.
- `approach`: how the paper attacks it, in 2-4 sentences: the key idea first, then the main components.
- `findings`: 2-6 key results, most important first, each one specific sentence. `section_ids` lists the ids of the sections (or captions, such as "tab2") that contain the evidence, so that a verifier can find it.
- `contributions`: what the paper adds (a method, system, dataset, benchmark, analysis or proof), one plain item each.
- `limitations`: limitations the authors state, and ones evident from the setup (for example: simulation only, one dataset, small scale). Leave it empty only if there are none.
- `methods`: named techniques, models, algorithms or tools the paper uses or proposes.
- `datasets`: datasets, benchmarks or workloads it evaluates on, by name.
- `keywords`: 3-8 lowercase topic keywords a researcher would search for.
