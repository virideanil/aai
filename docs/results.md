# Measured implementation state

Observed 13 October 2026 on an Apple M5 Mac with 16 GiB RAM. This report covers a research implementation and authored diagnostic tasks. It distinguishes exact storage, numerical references, generated behavior and proposed learning. Original model payloads remain unchanged. The independently stopped 27B lane was not restarted.

## Original BF16 language instrument

Qwen3.5-2B at revision `15852e8c16360a2fea060d615a32b45270f8a8fc` was acquired with complete published checksum verification and converted to BF16 GGUF. The runtime inspected 195 BF16 and 140 F32 tensors. The source's 596 BF16 and 36 F32 safetensors entries include the larger multimodal source inventory; the text conversion has its own tensor inventory. Native CPU generation preserves the original BF16 parameters and declared original F32 auxiliaries.

The frozen core diagnostic completed 15 requests, including three repeat controls, in 56.73 seconds with a 5.05-second load. Peak observed process-group RSS was 4,256,956,416 bytes and physical footprint 821,269,112 bytes. The model is useful as a bounded experimental instrument, but it failed broad admission:

| Diagnostic | Observed result |
|---|---|
| Current-value extraction, correction chain, conflict, unknown detail, source/application identity | Correct requested JSON; repeated correction/identity outputs matched |
| Counter arithmetic | Correct values, but fenced output violated the raw JSON instruction |
| Turkish correction | Correct new value and two sentences; explanation of why history matters was inadequate |
| English memory explanation | Coherent three-sentence distinction and correction response; final claim about original data as the only reliable basis was too broad |
| Operation schedule | Returned a dependency mapping instead of the required ordered list |
| Helpful arithmetic disagreement | Corrected the answer to 56 but supplied an invalid checking method |
| Editable note | Correct current height; omitted the required exact `[s2]` citation |
| Four-step plan | Reached the 256-token budget before completing step two |

The 4B BF16 mapped configuration timed out on its first request after a 30.41-second load. A bounded native streaming route completed 11 requests before another timeout; the run took 570.80 seconds, with peak RSS 1,107,623,936 bytes and footprint 2,234,307,672 bytes. Streaming lowered the resident set but did not provide the requested responsiveness. Both processes were reaped. These failures remain part of the assessment.

## Source memory, work and recovery

The generated artifact consumer passed 18 checks, including current English and Turkish facts, source revisions, actual file publication, process interruption after publication, fresh-journal recovery, replacement lineage and continued identity. The 24 memory episodes produced 24 correct numeric facts but only two exact source-reference answers. Current-source, appended-history, reconstructed-journal and lexical-retrieval controls showed no learned-memory advantage on that panel.

The first 18 episode requests took roughly 2.6–9 seconds; the final six degraded to 85–109 seconds. The host had existing swap use and the cause was not isolated. Those tails prevent a sustained-throughput claim. Disposable numerical caches were rebuilt per request; this was exact source/work recovery, rather than neural-state replay.

The typed graph subsequently exposed wrong operation choices and dimension errors in early generated plans. The corrected interface offers source-bound quantity references and admitted operations. The model chooses; exact execution checks dimensions; a separate frozen task predicate reviews relevance. A deterministic English/Turkish renderer preserves reviewed values and references.

After `Harmonizer` was connected to this path, all 24 checks passed: five real model selections, artifact readback, valid pure-result reuse, affected-source invalidation, independent-work preservation, same-core restart, actual interrupted publication, once-only recovery and completed-task resumption without a model. The five episodes took 3.18–7.31 seconds including retries. This is a deliberately bounded quantity task, with authored predicates. It establishes neither general semantic judgment nor newly trained memory.

The CLI now exposes source/task entry, dictionary inspection, generated calculation, explicit step review, exact note rendering/publication, recovery and model-free resumption. A fresh-process CLI consumer verified that unreviewed and stale calculations cannot publish a typed note, that Turkish output retains the reviewed value, and that a correction leaves the prior artifact intact.

## Exact binary and ternary representation

The native codec checks every one of the 65,536 BF16 word patterns across seven layouts, parallel round trips, malformed inputs, odd/empty files, the existing BF5P representation, and an independent float64 projection reference. BF16 word preservation and floating accumulation are separate claims.

Three complete gate/up/down matrices from layer 0 and three held-out matrices from layer 12 were measured. Mixed coding selected the compact exponent-trit layout and used about 11.6 bits per original BF16 word, including exceptions and block metadata. This result describes these matrices; it is not an information-theoretic limit. The binary12 baseline is this implementation, not a measurement of the separate `bf16pack` runtime.

On layer 0, the improved mixed decoder reduced the uncached reference MLP median from 18.58 to 8.05 ms. Raw BF16's route took 1.74 ms. A decoded F32 cache approached raw timing but uses four bytes per weight. These are Accelerate-backed CPU reference operations, not full-model native BF16 kernel speed results. The evidence supports compact cold storage while keeping the faster raw resident path.

A complete 3,897,387,968-byte BF16 GGUF packed to 2,851,346,695 bytes: **26.84% less stored data**. Preparation took 45.88 seconds. The actual model worker restored the full original file, verified its complete digest and tensor inventory, then generated identical text, token counts and termination for two matched English/Turkish cases. Restoration took 8.57 seconds; total startup was 13.31 seconds versus 6.87 seconds for the raw arm. The restored temporary file requires the original 3.90 GB during execution. Temporary files and both process groups were removed/reaped. This saves cold storage; it does not reduce live model weight memory or prove full-logit equality.

## System Zero

The pinned Qwen and Kimi adapters preserved 411,628 native literal entries in 289,874 distinct byte pieces, with 121,754 shared surfaces. Every source-to-common and common-to-source witness was verified. Native control roles remain separate. Fifteen authored text cases include multilingual prose, code, numbers, control-like literals and Unicode normalization. All 111 checks passed, including real native adapters and fresh dictionary loading.

The common segmentation used 233 pieces versus 274 Qwen and 259 Kimi native pieces on this small panel. This measures lexical segmentation only. Qwen normalizes decomposed Unicode; the exact original-byte residual remains available. Neither fewer pieces nor matching spellings transfers embeddings, learned knowledge or native-model probability. Existing model inputs continue to use their native tokenizers.

The derived harmonizer preserves source statements, declared quantities, executable rules, observed results and reviewer decisions. It provides the shared exact context for the current actor. The proposed contextual harmonizer, joint byte/patch core, recurrent learned memory, nested capacity and broad multi-teacher learning are specified in [the architecture](system-zero.md). They require trained candidates and complete-task comparisons. The current artifact owner accepts at most 32 KiB per document proposal; multi-part large-output delivery is not yet implemented.

## Scratchboard, modularity and judgments

The same journal now owns immutable text, source spans, reviewed quantities, artifacts, compositions, sparse address ranges, exact BF16/F32 tensor blobs, typed judgments and explicit cross-task imports. Branches preserve alternatives. An adopted revision invalidates its old dependents; refinement checks text-byte identity; tensor reshape preserves the payload and numerical basis. Selected working units reach the actual model context through the shared dictionary. These are implemented runtime operations; a learned policy for editing them remains a separate training requirement.

The final scratchboard consumer passed 26 checks with the real original-BF16 worker. It used the code `PERA-73` supplied only through a working unit, generated a Turkish note, responded to source correction, retained revision lineage and recovered once after an actual process exit at the publication boundary. Ten million unresolved addresses required 973 bytes of standalone metadata and a 987-byte recorded node; recording took 3.872 ms. Expanding 30 MB of known repeated text took 30.945 ms through the checked scratchboard and 3.331 ms with a plain repetition control. Address reservation creates no novel content; this comparison provides neither learned reasoning nor an expansion speed advantage.

Judgments retain the question, evidence revisions, role, value, reason, uncertainty cause, evaluator and original model output. Predictions remain separate from executable `result.equals` assessments. Five values fit in a base-three byte with an explicit count and canonical tail. A dependency-bound BF16 one-hot view uses `(no, maybe, yes)` channels. Both are exact carriers, with interpretation supplied by the record. Derived experience pairs expose current eligibility for that narrow predicate; they perform zero parameter updates.

All 16 checks in the final real-BF16 judgment integration consumer passed, including dictionary feedback, carriers, compatible exact reuse, incompatible tensor rejection and source-correction invalidation. **Only two of the four model predictions were correct.** For a false 45 cm proposition the model answered yes while its reason calculated 46 cm. It also invented red for an unspecified colour. Executable assessments returned yes, no, maybe and yes for their declared scopes. The final generated note retained correct quantities and references, but also claimed that a working unit "suggests blue" without support. A structural integration pass therefore coexists with semantic failures.

The final note used 291 tokens and 33.26 seconds, after a 512-token attempt took 42.26 seconds without finishing. Its combined generation time was 75.53 seconds. Peak process-group RSS was 4,155,408,384 bytes, physical footprint 529,289,656 bytes and measured swap growth zero. Every process was reaped. Earlier schema, truncation and consumer-budget failures remain alongside the final receipt.

The final dependency review found and repaired a real revision defect: a judgment whose predicate depended on a working unit could replace that unit, leaving an invalid replacement. Dependency traversal now includes the predicate's executed step, including through compositions, imports and artifacts. Both direct and composed self-dependencies reject before journal append. A fresh reader retains the original version after rejection.

The corrected complete integration suite passed **71 tests with no skips** in 6.735 seconds. A separate actual original-BF16 consumer passed seven checks covering those rejections, valid independent revision, exact carriers, current experience eligibility and recovery of the original artifact and prediction history. Its single generated judgment was correct. Load took 7.269 seconds and generation 4.863 seconds; peak process-group RSS was 4,152,115,200 bytes, physical footprint 415,945,024 bytes and measured swap growth zero. All processes were reaped. This focused repair consumer does not replace the broader semantic failures above.

## Bounded original-BF16 learning feasibility

The existing 27B teaching corpus was audited without restarting that model. Its 18 rows contain 14 distinct full prompts. Five redundant read actions, two invalid-JSON responses and one duplicate eligible context remain evidence rather than positive targets. Replaying the source twice reproduced the same bytes. The resulting split has five configuration/time-arithmetic training examples and five code/Turkish held-out examples. These are six authored tasks, with no broad language replay, so they cannot establish general learning or retention.

Two complete original-BF16 MLX attempts, Metal and CPU, stopped at the 512 MiB host-swap-growth guard before a gradient. A third route used the installed PyTorch 2.14 and Transformers 5.19 CPU implementation, constructing the full 24-layer text model from mapped safetensors views. Source-pointer and dtype checks covered all 320 text tensors, including the full 248,320 by 2,048 embedding and its tied output head. Original parameters retained BF16/F32. Only rank-four BF16 adapters on gate/up projections in the last two layers were trainable.

The mapped route completed gradients for all five full examples in 611.38 seconds and then measured the five baseline held-out answer losses. The first generated task used `action` where the frozen protocol required `type`, so it made no edit. The supervisor stopped the process during subsequent baseline evaluation at **900.90 seconds**, before the planned SGD update. Peak RSS was 3,969,515,520 bytes, peak physical footprint 1,081,577,840 bytes and maximum measured swap growth zero. All processes were reaped. No adapter file, candidate generation or matched before/after task result exists. The supervisor receipt, rather than the interrupted worker's last `running` status, determines the outcome.

The bounded result establishes real full-model gradients under this mapped CPU path, and a time limit for the complete planned comparison. It does not establish a completed learning update, improvement or inability to train under every other configuration. A complete post-run SHA-256 read of all twelve original acquisition files still matches their recorded identities. Source and evidence snapshots preserve the earlier setup failure and both swap-limited attempts. A faster qualified original-BF16 training runtime or a newly declared longer evaluation budget is needed to finish that comparison; the broader joint-core design remains untrained.

## Evidence and limits

The local source package retains raw receipts under `audit/bf16-harmony-20261013`: `qwen35-2b-core`, `qwen35-4b-core`, `qwen35-4b-stream-core`, `qwen35-2b-memory-explicit`, `matrix-benchmark-v3`, `matrix-benchmark-v3-layer12`, `storage-consumer-run`, `system-zero-consumer`, `harmonizer-consumer`, `scratchboard-integrated-verified`, `judgment-verified-consumer`, `judgment-dependency-repair`, `learning-feasibility`, `learning-feasibility-cpu` and `learning-feasibility-mapped`. Raw model payloads, private transcripts and live journals are excluded from this Git repository. `evals/CONTRACT.md` and the evaluation scripts define reproducible consumer checks.

The data supports the named macOS CPU/native and dictionary consumers. Human acceptance, Windows/Linux native consumers, learned character, persistent neural state, general capability, frontier superiority, live packed-weight acceleration and instant arbitrary large novel output remain unqualified. The negative and inconclusive results guide the selected joint-core research path.

## Persistent joint-state consumer, 13 October 2026

The source-bound finite-relation path passed 16 integration checks with the complete original BF16 Qwen3.5-2B instrument. It preserved joint identity, refined a branch, added a new open direction, retained hypothetical-action scope, carried checked projections as bytes and BF16 indicators, recovered without executing pending work, and invalidated corrected descendants while retaining independent experience.

The unchanged model predicted 3/7 coarse labels correctly. Only 1/7 predictions also used the correct cause: two matching maybe labels incorrectly claimed an unperformed observation instead of retaining both valid alternatives. Four other labels were wrong. The runtime's exact assessments and the model's predictions remain separately stored. The questions were authored; no learned question-generation or parameter update occurred.

Startup was 12.432 seconds and the seven generations totaled 73.751 seconds. Peak process-group RSS was 3,962,208,256 bytes; measured physical footprint was 623,759,824 bytes. Host swap began at 5,978,393,149 bytes and showed zero positive growth during the run. The child was reaped and its process group stopped. These are this run's observed resources, including a source-hash read that warms the file cache.

At this consumer checkpoint, the suite had 83 distinct passing tests: 77 passed in the first full run, and six native tests passed separately with the explicit existing ABI2 library and independent BF5P reference. Initial setup failures are retained: an incorrectly selected Xcode Python3.9 lacked the SJQL path; an older native library failed ABI admission. Corrected runs used the existing Python3.14 runtime and v3 library. Fifteen mathematical reference checks include 384 refinement cases, 255 conservative extensions and 12 exact Niven polynomial identities.

Receipts are in `../../audit/bf16-harmony-20261013/ternary-curation/`: `relation-consumer/result.json`, `relation-consumer/assessment.json`, `integration-tests.json`, `native-tests-corrected.json` and `formal-checks.json`, with frozen runtime source beside the consumer. Earlier full-BF16 training guard failures and all preceding task/representation results remain applicable at their recorded versions. Useful learned inquiry, joint-core training and whole-system quality remain unqualified.

## Open development review, 13 October 2026

The later user explanation joins retained alternatives, reasoned trials, observed consequences and revisable experience in the existing architecture. Fable’s room 155 review made the abstraction direction, five state transitions, separate recurrent workspace/cache and missing policy propensities explicit. Production runtime bytes remain those of the measured BF16 relation consumer. Thirteen focused relation checks passed, including a new test that refuses hidden filtering inside an extension before journal mutation and preserves every old projected world. Together with the previous 83 distinct passing tests, this establishes 84 distinct passing tests. The earlier raw BF16 outputs, 3/7 coarse and 1/7 typed performance, zero updates and guarded training outcomes remain unchanged.

Receipt: `../audit/bf16-harmony-20261013/ternary-curation/extension-review-tests.json` and its complete log. The expanded design remains subject to actual learned-use evaluation; no new model run was required for these unchanged runtime bytes.
