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

## Evidence and limits

The local source package retains raw receipts under `audit/bf16-harmony-20261013`: `qwen35-2b-core`, `qwen35-4b-core`, `qwen35-4b-stream-core`, `qwen35-2b-memory-explicit`, `matrix-benchmark-v3`, `matrix-benchmark-v3-layer12`, `storage-consumer-run`, `system-zero-consumer` and `harmonizer-consumer`. Raw model payloads, private transcripts and live journals are excluded from this Git repository. `evals/CONTRACT.md` and the evaluation scripts define reproducible consumer checks.

The data supports the named macOS CPU/native and dictionary consumers. Human acceptance, Windows/Linux native consumers, learned character, persistent neural state, general capability, frontier superiority, live packed-weight acceleration and instant arbitrary large novel output remain unqualified. The negative and inconclusive results guide the selected joint-core research path.
