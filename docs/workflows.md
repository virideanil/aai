# Native generation, work and learning workflows

These commands run from the repository root in a matching Python environment.
New journals use standard-library SQLite. Legacy SJQL journals require their
maintained package or `NANOAI_SJQL_SRC` pointing at its `src` directory.
Generation requires an already qualified native
model and server. The [README](../README.md) and [tokenizer guide](master-tokenizer.md)
provide a smaller model-free starting path.

## Admit an existing native source

`tools/prepare_native.py` verifies the actual GGUF tensor inventory and pins the model, server, supplied libraries and runtime sources. Example, from this repository:

```sh
python tools/prepare_native.py --model /absolute/path/text-bf16.gguf \
  --server /absolute/path/llama-server \
  --library /absolute/path/libllama.dylib \
  --library /absolute/path/libggml.dylib \
  --output .local/core
```

Supply every native dependent library for the chosen build. The resulting `.local/core/core.json` is machine-local configuration. Existing payloads stay unchanged. Streaming and optional packed storage have explicit profiles; read the measured costs before selecting them.

`--backend cpu` preserves the CPU control. The experimental `--backend metal`
selects the native `MTL0` device with all model layers requested on Metal and CPU
fallback for unsupported operations. It requires `--weight-access mapped` and
keeps the original BF16 model and BF16 caches. The native log records actual
placement. On the measured 16 GiB host, both qualification attempts stopped at the
host swap-growth guard during startup. No output, speedup or numerical-equivalence
result qualified this option for automatic selection.

Preparation also registers the actual tensor inventory in
`.local/core/tensors.sqlite3`. `--catalogue PATH` selects an existing owned
catalogue for sharing across profiles. The native reader binds the immutable
version and checks its tensor names, shapes, types and byte extents against the
verified payload. Other runs can append to the database while this version stays
bound. Source GGUF and candidate safetensors files retain the tensor values.

Thinking is enabled in every native generation request. The default token limit
and request deadline are `None`; the model ends naturally within its declared
context capacity. Memory guards and cancellation still apply. The default
sampling profile is an explicit greedy control. `--sampling qwen35-thinking`
selects the retained model card's candidate text-thinking settings. A profile
records each numeric parameter and seed; traced responses also check the applied
native settings. Select a profile from measured task behavior and retention.

For example, SQL can inspect the prepared inventory directly:

```sh
python - <<'PY'
from contextlib import closing
import sqlite3

with closing(sqlite3.connect(".local/core/tensors.sqlite3")) as db:
    print(db.execute("SELECT dtype, COUNT(*), SUM(elements) FROM tensors GROUP BY dtype").fetchall())
PY
```

## Work, review and recovery

Run from the repository with `PYTHONPATH=src`. Use a fresh private state directory. Put source facts in a UTF-8 file; for calculations, quantities have explicit values and units, for example `{"length":{"value":13,"unit":"cm"},"count":{"value":4,"unit":"count"}}`.

```sh
python -m nanoai.cli --state .local/work init
python -m nanoai.cli --state .local/work source survey /absolute/path/survey.json
python -m nanoai.cli --state .local/work task report "Calculate the total length" --filename report.md
python -m nanoai.cli --state .local/work dictionary report --source survey
python -m nanoai.cli --state .local/work calculate report --source survey --core-config .local/core/core.json
```

Inspect the returned operation, source fields, units, result and `step_id` against the task. Record that assessment explicitly. Replace `STEP_ID` below with the returned ID and give the actual reviewer/reason:

```sh
python -m nanoai.cli --state .local/work review-step STEP_ID confirmed --reviewer local-user --reason "The selected length and count answer this task"
python -m nanoai.cli --state .local/work note STEP_ID --language en --commit
python -m nanoai.cli --state .local/work recover
python -m nanoai.cli --state .local/work resume report
```

`note` renders the reviewed calculation without a model call. It publishes only with `--commit`. `run report --core-config ...` instead asks the model for a document proposal, optionally bound to `--observation STEP_ID`; add `--commit` to publish it locally. A valid citation does not establish unrestricted factual correctness. Correct a source using `source survey FILE --expected-revision 1`; affected completed work reopens, and earlier artifacts remain available.

System Zero's optional lexicon is prepared from explicit source manifests with `tools/prepare_syszero.py --source MANIFEST --source MANIFEST --out .local/system-zero`. Native adapters retain source hashes, model namespaces and normalization residuals. Common segmentation is a lexical reference and is never silently substituted for a native model's input tokenizer.

The scratchboard holds exact text/source spans, reviewed quantities, artifacts, compositions, unresolved ranges and BF16/F32 tensor blobs. Its nodes share the journal and dependency checks. `note` already uses this path. To inspect or pass a reviewed node to the model, replace `NODE_ID` with the ID returned by `board-note`:

```sh
python -m nanoai.cli --state .local/work board-note STEP_ID --language tr
python -m nanoai.cli --state .local/work dictionary report --source survey --board NODE_ID
python -m nanoai.cli --state .local/work run report --source survey --board NODE_ID --core-config .local/core/core.json
python -m nanoai.cli --state .local/work board-branch NODE_ID --reason "Keep this candidate for comparison"
```

`board-reserve`, `board-bind`, `board-add`, `board-revise`, `board-refine`, `board-reshape` and `board-latest` expose the structural operations. Their `--help` lists inputs. A revision preserves history and invalidates dependent work; exact refinement and tensor reshape enforce their declared invariants. The [scratchboard contract](scratchboard.md) defines bounds, basis compatibility and the distinction between structural modularity and a learned editing policy.

`judge`, `assess-judgment`, `carry-judgments` and `experiences` expose provisional judgments, independent numerical or joint-relation property checks, exact ternary/BF16 carriers and attributed feedback. A model can choose the wrong label or cause; the original prediction remains visible. Recording or packing an experience does not train the model. The `step` command also exposes admitted pure relation queries, conditional refinement, conservative extension and hypothetical state maps through JSON arguments. The [relation contract](relations.md) defines the input and limits. `board-import` supports explicit cross-task reuse with current source dependencies and a recorded relevance judgment. Exact units retain provenance across core versions; tensor-dependent units require a compatible basis.

## Developmental learning experiment

The [frozen comparison](../evals/developmental_contract.md) connects full original-model features and UTF-8 bytes to a jointly trained recurrent selection network, scoped evidence judgments, actual document choices and independent artifact assessment. The pretrained language backbone stays unchanged. Training consumes current checked experience from the same work journal, including unsuccessful choices with their original observations. The authored menu supplies replayable alternative targets separately from the chosen action's observed consequence.

`tools/encode_developmental.py` collects source-pinned features. `tools/evaluate_developmental.py` exposes collection, training and artifact phases. `tools/evaluate_developmental_native.py` runs the unchanged full-BF16 control. `tools/evaluate_developmental_recovery.py` checks fresh candidate/workspace restoration, real process interruption at publication and once-only recovery. Their `--help` lists required manifests, fresh output directories and sourced launch preflights. The recorded candidate and original model sources retain separate identities.

For an existing experiment state, inspect recorded outcomes and registered candidates with:

```sh
python -m nanoai.cli --state /absolute/path/experiment-state experiences TASK_ID
python -m nanoai.cli --state /absolute/path/experiment-state learners
```

The experience view keeps judgment-derived and action-derived records separate. It shows current eligibility and retained historical assessments without feeding evaluator targets into model prompts. Inspection changes no parameter or admission state.

`trial-act TASK_ID --source SOURCE_ID --candidate CANDIDATE_ID --features FEATURE_DIRECTORY --choices CHOICES_JSON` executes a registered experimental candidate and publishes its chosen local document under the task's existing grant. The source contains `{"observations":["..."]}` and choices follow the frozen contract's `id`, `kind` and `content` fields. Every question, observation and choice text needs an exact feature-cache entry. The command identifies the experimental mode and admission status; it does not select a new default model. This menu consumer stays separate from the full-source generative memory experiment below.

## Generative memory and learning controls

`tools/evaluate_recurrent_credit.py` uses the complete local source model and its native recurrent cache. It takes a current checked experience, isolates an earlier recurrent write as the route for a later answer loss, changes a separately saved BF16 adapter, generates and publishes an actual local answer, then saves the complete hybrid state. `--reload-only` verifies the saved candidate and state in a fresh process, checks identical continuation, and exercises source correction and once-only recovery in that private experiment state. Use its `--help` for explicit source, experience, collection-state, output, source-manifest and launch-preflight inputs. Keep the model source unchanged and run under the documented resource guards.

`tools/evaluate_outcome_learning.py` compares gradients from the actual chosen action's assessed success with the original rubric-supervised recipe. It records validation-selected and fixed final-epoch results, reproduces the original selected candidates, and publishes the new experimental candidates' actual documents. Training and validation information budgets are explicit in the [developmental contract](../evals/developmental_contract.md). These tools retain failed candidates and do not change the default model. [Measured results](results.md) describe the qualified scope.

## Writing and applicable teaching

`write-phase` takes a UTF-8 instruction file, current source IDs and a native core
configuration. It records the actual prompt, sampled IDs, policy, generated
artifact and source dependencies. It also observes the generated text in the
language catalogue. Repeating a completed phase reuses the observed effect.
`assemble` joins selected current generated sources into an editable local
manuscript. The [writing contract](agentic-writing.md) distinguishes execution,
brief compliance, causal coherence, literary judgment and parameter learning.

`lesson-install` accepts an authored lesson with a principle, scope, reason,
worked example, contrast and next check. `teaching-feedback` binds attributed
feedback to quotations from an actual draft and its current evidence.
`teaching-pair` exports a current, checked training correction while retaining
the original failure. `Writing.practice` executes a declared number of complete
attempts, checks each observed artifact, and uses actionable feedback for the
next revision. It retains all attempts and selects a result only when the scoped
rubric passes. This process alone performs no parameter updates.

`tools/evaluate_native_teaching.py` collects actual attempts and measures original,
guided, adapted and recovered native consumers under the same protocol.
`tools/evaluate_teaching.py --stage train --collection PATH` trains from the checked
collection plus authored demonstrations and replay, with validation used for
selection. `tools/export_native_adapter.py` preserves the selected BF16 adapter
coefficients in the qualified native layout. Tensor versions, parent links and
learning events are queryable in the explicitly selected SQLite catalogue.
Each script lists its required frozen inputs with `--help`; failed or incomplete
stages retain their evidence and grant no candidate admission.
