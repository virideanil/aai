# Computation, memory and continuing agency

Status: integrated research design, 13 October 2026. The governing joint-core synthesis is [System Zero and Quantum tokening](system-zero.md); [measured results](results.md) distinguish the working foundation from proposed neural computation.

## Purpose

nano'ai should understand unfamiliar work, reason and create, act within entrusted scope, inspect consequences and learn from correction. It retains identity, history and commitments across parameter and runtime changes. The research ambition includes exceeding strong existing systems in useful capability and efficiency. Each claimed improvement needs a named comparator, matched inputs and costs.

The current investigation studies how Qwen and Kimi accomplish their computation. The earlier crop-initialization experiment was an assistant-selected method; its results do not answer this architectural question. Original model artifacts remain unchanged. A resident pretrained baseline helps measure our mechanisms without making its architecture the final design.

## What the source implementations teach

| Mechanism | Source observation | Design consequence for nano'ai |
|---|---|---|
| Gated delta memory | Qwen3.8 uses the Qwen3.5 hybrid architecture. Its recurrent module decays an associative matrix, predicts a value from a key and writes the prediction error. Full attention remains in selected layers. | Separate compressed recurrent state from recoverable detail. Evaluate forgetting, interference and correction, as well as ordinary recall. |
| Kimi Delta Attention | K3 learns channel-dependent decay and write strength, with short-convolution history. Its reference cache preserves recurrent, convolution and attention states. | A restorable episode must include every state component. A corrected earlier observation requires valid replay or a separately proven update rule. |
| Attention Residuals | K3 weights normalized representations from earlier depth blocks rather than always summing every contribution equally. | Learned selection can operate over prior computation as well as over input tokens. Test selection against a strong fixed policy before adopting it. |
| Latent MoE | K3 routes to a subset of experts in a learned lower-dimensional space and adds shared computation. | Conditional capacity saves active work only when routing, resident experts and movement are included in cost. Expert count alone gives no laptop advantage. |
| MTP and DSpark | Prediction heads propose future tokens. The acquired K3 DSpark uses several target hidden states and requires target verification. | Study multi-step learning and useful proposals separately from independent reasoning. Measure total outcome time and rejected work. |
| Quantization-aware learning | K3's released expert format and BitNet's ternary computation pair learned weights with particular representations and consumers. | Train new ternary operators under their actual forward rule; choose storage and execution from end-to-end cost. |

Sources: [Qwen3.8 model account](https://huggingface.co/Qwen/Qwen3.8-27B), [pinned K3 implementation](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py), [K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json), [DSpark configuration](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark/blob/38a88101e0d46bb22134b9da340f381b954d40d4/config.json). Exact inspected local versions are retained in the local audit.

For one recurrent head, a useful abstract form is:

```text
decayed = decay * memory
error = value - decayed @ key
memory_next = decayed + write_strength * outer(error, key)
output = memory_next @ query
```

This computation learns what to retain and overwrite. Its latent mixture does not by itself provide an exact source revision, a retracted claim or permission to act. nano'ai needs both trained representations and those explicit relationships.

## Selected experiment and alternatives

The first question is whether a bounded memory can select useful valid evidence, preserve unfinished work and recover after interruption while a capable language core retains its useful behavior. Compare an ordinary recent-context core, exact revision-aware retrieval, and a learned selector where training produces evidence of benefit. Measure the complete generated task, rather than only a retrieval score.

The durable source memory uses the maintained SJQL transaction journal. A task records its goal, current revision, source versions and identity. Learned selection ranks eligible evidence; exact revision and access rules define eligibility. The model receives source text as data with explicit citations. Model proposals carry their dependency snapshot and cannot commit after that snapshot changes. A selection records its policy, parameters and ordered output identity. Validation re-runs that policy: an unrelated insertion that leaves its output unchanged remains compatible, while changed selected evidence or a newly selected source invalidates the pending proposal. Explicit source selection retains only its named dependencies. Historical epoch-based snapshots keep their original conservative validation law.

An output becomes a durable local artifact through a create-only prepared operation, followed by byte readback. Recovery observes the actual file before retrying. A crash after publication and before receipt recording is reconciled without repeating the effect. Changed sources invalidate dependent proposals while retaining unrelated completed artifacts. A replacement operation records the prior operation in `supersedes`; previous bytes remain available. Recovery publishes already fixed journal content, and its receipt records `core_identity_at_proposal`, even if a newer core is now available. Source citations and factual correctness have separate checks in the artifact evaluation.

The selected learned destination is one jointly trained byte/patch core with recurrent workspace, nested BF16 capacity and action heads. Source-selection experiments and the existing core/retrieval consumer remain controls. Each trained selector or workspace must improve complete generated tasks through correction and distraction at matched cost. The exact dictionary is a source/rule view of the journal; it supplies context and observed outcomes without claiming semantic learning.

## Continuing identity and education

The identity record holds the continuing name, lineage, commitments and development history. Learned dispositions and expressive habits require education and observed behavior. Preserve Gi, Yu, Jin, Rei, Makoto, Meiyo and Chugi, together with honesty, loyalty, wisdom through experience and care for wellbeing and agency.

Education must include useful work, uncertainty, mistaken teachers, inconvenient errors, constructive disagreement and repair. General language replay and held-out conduct cases accompany parameter updates. Neither an identity file nor a preferred-sentence score establishes character.

## Broader architecture retained

Quantum refinement chooses useful source detail and internal effort. Matryoshka modes require jointly trained consumers. Hypertensoring composes compatible learned and exact operators. Binary hardware carries ternary projections with explicit scales; activation, cache and accumulator precision remain independent choices. Perception, editable creation, expressive interaction and independently identified cooperating agents retain their own consumers.

The resident instrument now requires original BF16 weights. Original FP32 vectors and recurrent state remain at their declared precision. The client rejects older configurations before starting a process; the native adapter inspects actual GGUF tensor types and rejects F16, Q4 and other quantized storage. Exact binary or ternary representations must reconstruct the original BF16 words and preserve the consuming arithmetic contract. Standalone trained ternary components retain a separate research identity and require an explicit admission decision before serving in this BF16 lane.

The laptop evaluation and existing original-BF16 target retain separate model identities and runtime contracts. The independently stopped large-model execution lane stays at its existing boundary.



## Connected current implementation

`syszero.py` imports explicit native tokenizer manifests, verifies source identities and preserves exact bytes with namespaced witnesses and normalization residuals. `harmonizer.py` derives the task's current evidence, declared quantities, operators and observed results from the same journal. It recomputes and validates packet identity after a source, review or implementation change.

`agent.calculate` supplies current quantity references to the real BF16 generator, then compiles its selected operation into `steps.py`. The pure DAG records source reads, field selection and integer/dimensioned arithmetic. Operator execution, semantic task review and external effect authority have separate meanings. `documents.py` creates exact English/Turkish notes only from confirmed current calculations. The CLI exposes these operations, review, publication, recovery and model-free completed-task resumption.

The native client starts a dedicated process group and closes it on timeout, cancellation, protocol error or worker exit, including descendants whose parent exits first. The worker uses a private Unix socket. Runtime profiles explicitly control mapped/streamed source access, BF16 admission and optional exact cold-file restoration. Every resource guard observes its configured process group; RSS, physical footprint and host swap remain separately reported.

The core's future learning record includes complete context, candidate/model identity, original outputs, task family, versioned sources, independent verdicts and actual effects. Data splits hold out source/task families; repeated turns stay together. Teacher agreement is a prediction metric. General language, creation and conduct require diverse replay and representative evaluation beyond the present quantity tasks.
