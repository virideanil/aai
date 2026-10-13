# Frozen vocabulary sources

This tokenizer is derived from the 22 pinned payloads below. The
[manifest](manifest.json) binds each complete payload identity to the cold
inventory, selection policy and final byte artifact. Source membership, native
IDs and role labels remain in the separately supplied cold inventory.

The table reproduces the inspected source declarations. Exact license files
already captured in the local source audit are included unchanged where
available, with full identities in [the file record](source-license-records.json).
Some sources expose a metadata declaration without a captured license file;
custom declarations remain custom. The captured Nemotron LICENSE is empty and
its metadata says `other`. No umbrella license has been assigned to the derived
artifact. This record supplies attribution and preserves the distinct source
terms for later distribution decisions.

| Source | Pinned version | Recorded declaration | Captured file |
|---|---|---|---|
| [gpt2](https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | mit | Metadata declaration only |
| [cl100k_base](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken) | Full payload SHAKE256 in manifest | see shared tiktoken source-license evidence | [tiktoken/LICENSE](source-licenses/tiktoken/LICENSE) |
| [o200k_base](https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken) | Full payload SHAKE256 in manifest | see shared tiktoken source-license evidence | [tiktoken/LICENSE](source-licenses/tiktoken/LICENSE) |
| [deepseek-v3](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json) | `e815299b0bcbac849fa540c768ef21845365c9eb` | No short metadata label; see captured files | [deepseek-v3/LICENSE-CODE](source-licenses/deepseek-v3/LICENSE-CODE), [deepseek-v3/LICENSE-MODEL](source-licenses/deepseek-v3/LICENSE-MODEL) |
| [bloom](https://huggingface.co/bigscience/bloom/resolve/7f10a99ce7c08f03c7719a586cb2cbda1433ac05/tokenizer.json) | `7f10a99ce7c08f03c7719a586cb2cbda1433ac05` | bigscience-bloom-rail-1.0 | Metadata declaration only |
| [falcon](https://huggingface.co/tiiuae/falcon-7b/resolve/ec89142b67d748a1865ea4451372db8313ada0d8/tokenizer.json) | `ec89142b67d748a1865ea4451372db8313ada0d8` | apache-2.0 | Metadata declaration only |
| [olmo-2](https://huggingface.co/allenai/OLMo-2-1124-7B/resolve/7df9a82518afdecae4e8c026b27adccc8c1f0032/tokenizer.json) | `7df9a82518afdecae4e8c026b27adccc8c1f0032` | apache-2.0 | Metadata declaration only |
| [smollm3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/a07cc9a04f16550a088caea529712d1d335b0ac1/tokenizer.json) | `a07cc9a04f16550a088caea529712d1d335b0ac1` | apache-2.0 | Metadata declaration only |
| [granite-3.3](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct/resolve/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b/tokenizer.json) | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | apache-2.0 | Metadata declaration only |
| [mt5](https://huggingface.co/google/mt5-base/resolve/2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f/spiece.model) | `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f` | apache-2.0 | Metadata declaration only |
| [xlm-roberta](https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/sentencepiece.bpe.model) | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | mit | Metadata declaration only |
| [nllb-200](https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/f8d333a098d19b4fd9a8b18f94170487ad3f821d/sentencepiece.bpe.model) | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` | cc-by-nc-4.0 | Metadata declaration only |
| [bert-turkish](https://huggingface.co/dbmdz/bert-base-turkish-cased/resolve/b6e1de16c983e0f2c70664591ea3f22810072608/vocab.txt) | `b6e1de16c983e0f2c70664591ea3f22810072608` | mit | Metadata declaration only |
| [mistral-nemo](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/resolve/a4477a2f977929a969745b69bbd62e03043551a5/tokenizer.json) | `a4477a2f977929a969745b69bbd62e03043551a5` | apache-2.0 | Metadata declaration only |
| [phi-4](https://huggingface.co/microsoft/phi-4/resolve/2db69c1c3e91a05d2c64a3185acfbaf36f744e25/tokenizer.json) | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | mit | [phi-4/LICENSE](source-licenses/phi-4/LICENSE) |
| [yi-1.5](https://huggingface.co/01-ai/Yi-1.5-9B/resolve/80d5471b1eae28beae33e06eadbd4b48e74d4ce1/tokenizer.json) | `80d5471b1eae28beae33e06eadbd4b48e74d4ce1` | apache-2.0 | Metadata declaration only |
| [exaone-3.5](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/resolve/553ea250b9a5317231459279d5847d6cf955b9aa/tokenizer.json) | `553ea250b9a5317231459279d5847d6cf955b9aa` | other | [exaone-3.5/LICENSE](source-licenses/exaone-3.5/LICENSE) |
| [nemotron-mini](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct/resolve/791833e92ebddb0bc2c1007f6d2b6764f886a2ae/tokenizer.json) | `791833e92ebddb0bc2c1007f6d2b6764f886a2ae` | other | [nemotron-mini/LICENSE](source-licenses/nemotron-mini/LICENSE) |
| [kimi-k2](https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/tiktoken.model) | `fd1984e2b7a3350dbf7305fe73a4ede25c14de50` | other | [kimi-k2/LICENSE](source-licenses/kimi-k2/LICENSE) |
| [gpt-oss](https://huggingface.co/openai/gpt-oss-20b/resolve/6cee5e81ee83917806bbde320786a8fb61efebee/tokenizer.json) | `6cee5e81ee83917806bbde320786a8fb61efebee` | apache-2.0 | [gpt-oss/LICENSE](source-licenses/gpt-oss/LICENSE) |
| [qwen3.8-27b](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | apache-2.0 | [qwen3.8-27b/LICENSE](source-licenses/qwen3.8-27b/LICENSE) |
| [kimi-k3](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06) | `f831ab66814297da540d832a5235f8e904f29d06` | Original K3 source file retained | [kimi-k3/LICENSE](source-licenses/kimi-k3/LICENSE) |

The two tiktoken blobs are pinned by content identity. The captured source
implementation at `4e71bbe0c078468e00fefbf94b39849389f346e5` lists their expected
SHA-256 checksums; both complete local blobs matched. Their shared source
LICENSE is preserved above. Those upstream checksum formats retain their
original algorithm.

Only the bounded derived vocabulary and its provenance are included here.
Original complete tokenizer payloads, model weights, private calibration records
and live journals remain outside this Git repository. Executed source behavior
and its normalization residuals are recorded in the parent project's
`audit/tokenizer-finalization-20261013/native-sources-v1`.
