# nano'ai

nano'ai develops a continuing learning agent that understands, reasons, creates, acts and learns from correction. The selected architecture connects **System Zero's exact knowledge/representation contracts, one jointly trained core, and an exact work graph**. Quantum patching, recurrent memory, nested BF16 capacity and action heads share that core's learning objective. Identity, character, perception, editable creation, hardware efficiency and the wider requirements retain their scope.

This repository implements the current research foundation: original-BF16 model admission, native generation, versioned evidence/task memory, typed operations, a common vocabulary bridge, dictionary views, local artifact publication and observed recovery. The new joint neural core is a design and training research direction. Existing Qwen models supply complete experimental instruments; Qwen/Kimi mechanisms inform our own architecture.

- [Complete System Zero and joint-core design](docs/system-zero.md)
- [State, learning and execution contracts](docs/design.md)
- [Measured results and retained failures](docs/results.md)
- [Evaluation contract](evals/CONTRACT.md)

## Dependencies

Use Python 3.14 or newer and the maintained SJQL package. Set `NANOAI_SJQL_SRC` to its `src` directory if it is not installed. The source tree is usable with `PYTHONPATH=src`; `python -m pip install -e .` adds the `nanoai` command. Vocabulary adapters additionally use the pinned `syszero` extra. The exact journal and core process client use the Python standard library; evaluation/benchmark scripts also need NumPy and safetensors where used.

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

## Verification

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python tools/evaluate_core.py --config .local/core/core.json --out .local/core-evaluation
python tools/evaluate_steps.py --config .local/core/core.json --panel evals/quantity-panel.json --mode typed-quantities --out .local/quantity-evaluation
```

`NANOAI_HARMONY_LIB` selects the built native codec for its tests. `NANOAI_BF5P_REFERENCE` selects the existing independent BF5P encoder; tests report skips when these optional native dependencies are absent. Use one model process at a time and retain the configured request, footprint and swap guards. Full evaluation paths are documented by each script's `--help` and the [contract](evals/CONTRACT.md).

Model payloads, private conversations, candidate adapters and live journals remain outside Git. No external service or automatic model admission is part of these commands. Configuration carries native dependency paths explicitly.
