# nano'ai

nano'ai develops a continuing learning agent that understands, reasons, creates, acts and learns from correction. The selected architecture connects **System Zero's exact knowledge/representation contracts, one jointly trained core, and an exact work graph**. Quantum patching, recurrent memory, nested BF16 capacity and action heads share that core's learning objective. Identity, character, perception, editable creation, hardware efficiency and the wider requirements retain their scope.

The [whole architecture](../NANOAI-ARCHITECTURE.md) distinguishes the current generative instrument, bounded feature learner and destination byte/patch core. The preferred learning path joins generative memory and actions, then introduces chunking and capacity with matched controls. The current small selection experiment produced real parameter updates and artifacts but no reliable gain over its lexical control; no candidate was admitted.

This repository implements the current research foundation: original-BF16 model admission, native generation, versioned evidence/task memory, typed operations, a common vocabulary bridge, modular working representations, judgments and checked experience, local artifact publication and observed recovery. The new joint neural core is a design and training research direction. Existing Qwen models supply complete experimental instruments; Qwen/Kimi mechanisms inform our own architecture.

- [Complete System Zero and joint-core design](docs/system-zero.md)
- [Modular scratchboard and revision contracts](docs/scratchboard.md)
- [Persistent joint alternatives and their queries](docs/relations.md)
- [State, learning and execution contracts](docs/design.md)
- [Measured results and retained failures](docs/results.md)
- [Evaluation contract](evals/CONTRACT.md)

## Dependencies

Use Python 3.14 or newer and the maintained SJQL package. Set `NANOAI_SJQL_SRC` to its `src` directory if it is not installed. The source tree is usable with `PYTHONPATH=src`; `python -m pip install -e .` adds the `nanoai` command. Vocabulary adapters additionally use the pinned `syszero` extra. The exact journal and core process client use the Python standard library; evaluation/benchmark scripts also need NumPy and safetensors where used.

The optional `learning` extra pins the NumPy and PyTorch versions used for the F32 candidate. The `generative` extra pins the qualified CPU PyTorch/Transformers source adapter and tokenizer. Reuse a matching installed environment when available. Model payloads and feature caches have separate source manifests.

Native generation requires an existing qualified `llama-server` build that supports Qwen3.5 and original BF16, plus its dependent libraries. Native codec compilation currently uses C++17 and macOS Accelerate; Linux/Windows compilation has not been qualified. `tools/build_harmony.py --help` gives the explicit compiler/output options. The optional codec is independent of ordinary raw-BF16 generation.

`tools/prepare_native.py` verifies the actual GGUF tensor inventory and pins the model, server, supplied libraries and runtime sources. Example, from this repository:

```sh
python tools/prepare_native.py --model /absolute/path/text-bf16.gguf \
  --server /absolute/path/llama-server \
  --library /absolute/path/libllama.dylib \
  --library /absolute/path/libggml.dylib \
  --output .local/core
```

Supply every native dependent library for the chosen build. The resulting `.local/core/core.json` is machine-local configuration. Existing payloads stay unchanged. Streaming and optional packed storage have explicit profiles; read the measured costs before selecting them.

## Work, review and recovery

Run from the repository with `PYTHONPATH=src` and `NANOAI_SJQL_SRC` configured. Use a fresh private state directory. Put source facts in a UTF-8 file; for calculations, quantities have explicit values and units, for example `{"length":{"value":13,"unit":"cm"},"count":{"value":4,"unit":"count"}}`.

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

`board-reserve`, `board-bind`, `board-add`, `board-revise`, `board-refine`, `board-reshape` and `board-latest` expose the structural operations. Their `--help` lists inputs. A revision preserves history and invalidates dependent work; exact refinement and tensor reshape enforce their declared invariants. The [scratchboard contract](docs/scratchboard.md) defines bounds, basis compatibility and the distinction between structural modularity and a learned editing policy.

`judge`, `assess-judgment`, `carry-judgments` and `experiences` expose provisional judgments, independent numerical or joint-relation property checks, exact ternary/BF16 carriers and attributed feedback. A model can choose the wrong label or cause; the original prediction remains visible. Recording or packing an experience does not train the model. The `step` command also exposes admitted pure relation queries, conditional refinement, conservative extension and hypothetical state maps through JSON arguments. The [relation contract](docs/relations.md) defines the input and limits. `board-import` supports explicit cross-task reuse with current source dependencies and a recorded relevance judgment. Exact units retain provenance across core versions; tensor-dependent units require a compatible basis.

## Developmental learning experiment

The [frozen comparison](evals/developmental_contract.md) connects full original-model features and UTF-8 bytes to a jointly trained recurrent selection network, scoped evidence judgments, actual document choices and independent artifact assessment. The pretrained language backbone stays unchanged. Training consumes current checked experience from the same SJQL journal, including unsuccessful choices with their original observations. The authored menu supplies replayable alternative targets separately from the chosen action's observed consequence.

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

`tools/evaluate_outcome_learning.py` compares gradients from the actual chosen action's assessed success with the original rubric-supervised recipe. It records validation-selected and fixed final-epoch results, reproduces the original selected candidates, and publishes the new experimental candidates' actual documents. Training and validation information budgets are explicit in the [developmental contract](evals/developmental_contract.md). These tools retain failed candidates and do not change the default model. [Measured results](docs/results.md) describe the qualified scope.

## Verification

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python tools/evaluate_core.py --config .local/core/core.json --out .local/core-evaluation
python tools/evaluate_steps.py --config .local/core/core.json --panel evals/quantity-panel.json --mode typed-quantities --out .local/quantity-evaluation
python tools/evaluate_scratchboard.py --config .local/core/core.json --out .local/scratchboard-evaluation
python tools/evaluate_judgments.py --config .local/core/core.json --out .local/judgment-evaluation
```

`NANOAI_HARMONY_LIB` selects the built native codec for its tests. `NANOAI_BF5P_REFERENCE` selects the existing independent BF5P encoder; tests report skips when these optional native dependencies are absent. Use one model process at a time and retain the configured request, footprint and swap guards. Full evaluation paths are documented by each script's `--help` and the [contract](evals/CONTRACT.md).

Model payloads, private conversations, candidate adapters and live journals remain outside Git. No external service or automatic model admission is part of these commands. Configuration carries native dependency paths explicitly.
