# Persistent joint alternatives

A state carries the alternatives and their shared variable identities. A visible no/yes/both result is the image of a particular question over that state. This lets an open state survive a restart, acquire a condition, gain a new variable or support several questions while retaining its source and history.

For example, `x` and `y` are each open in both `[[0,0],[1,1]]` and `[[0,1],[1,0]]`. Equality is always true in the first relation and always false in the second. Two isolated maybe labels lose that distinction. The complete design connects this state to learning, action, identity and creative work; the current finite executor is one bounded implementation of the representation contract.

## Input and operations

An exact UTF-8 source can contain:

```json
{"type":"finite_relation_v1","variables":["x","y"],"worlds":[[0,0],[1,1]],"scope":"Declared linked switches"}
```

`Steps` records each operation, parent observation and source revision in the existing SJQL journal. Use the `step` CLI command with a JSON arguments file, or the `Steps.propose` and `Steps.execute` API. A source read supplies the parent for decoding; each returned `step_id` supplies the next parent.

| Operator | Arguments besides `step` | Meaning |
|---|---|---|
| `relation.decode` | none | Validate an exact JSON source as a finite model |
| `relation.query` | `expression` | Compute all possible answers and keep witnesses |
| `relation.refine` | `expression`, `expected` | Retain alternatives satisfying the explicit bit premise |
| `relation.extend` | `variable`, `expression` | Add a fresh variable; null opens both values, an expression derives its value |
| `relation.map` | `variable`, `expression` | Apply a hypothetical state transformation; external effect is none |

Expressions are 0 or 1, a variable name, `{"not":expr}`, or a two-element `and`, `or`, `xor` or `eq` junction. Each expression is evaluated against one shared assignment. Repeated names preserve identity. The current bounds are 12 variables, 4096 worlds, 128 expression nodes and depth 16. Resource-limit rejection remains distinct from a logical answer.

For a loaded relation step, an arguments file may contain `{"step":"RETURNED_STEP_ID","expression":{"eq":["x","y"]}}`. The command is:

```sh
python -m nanoai.cli --state .local/work step report relation.query query.json --source relation
```

Use the actual task, source and returned step identities. The source must be in the explicit current selection. The CLI records a failed admitted execution as a rejected step. Unexecuted proposals remain pending during recovery.

## Judgment, revision and learning

The query keeps its possible outcomes, their witnesses, variable names, expression and model scope. An empty relation records conflict and has no checked ternary image. A nonempty relation's projection can be assessed with `{"operator":"relation.image","step_id":"QUERY_STEP_ID","expected":null}`. `Judgments` records the model prediction separately from the executed assessment. A checked two-outcome image has cause `both_alternatives`; an unperformed check has cause `awaiting_observation`.

Current exact images can supply scoped training targets. The record retains the original model prompt, response, source versions and outcome. Source correction, rejected premises, changed executor versions and incompatible numerical dependencies invalidate affected current examples. Historical examples remain available with their original status. Creating such records performs zero parameter updates.

The user's middle arrows support both revision and encoding. Revision keeps the previous state and its change reason. Binary packing and BF16 one-hot indicators preserve the referenced judgment; the relation, witnesses and provenance remain in the graph. Packed label size is not the total state size.

The developmental contract treats each attempt as an experience with a purpose, alternatives, selected action, observed outcome and a reason for the next change. A scoped assessment can settle a question while leaving the continuing system open to new evidence and directions. This joins holding alternatives with a revisable choice, the user's superposition and humble-position analogy. Progress is assessed through task benefit, new understanding and retained capabilities.

An extension preserves every previous joint alternative. Its new expression is total on all old assignments; a filtering condition belongs to a separately recorded `relation.refine`. The present runtime rejects resource overflow explicitly. A future approximate representation must name its abstraction and preserve the inclusion of all old alternatives in its concretization, for the questions whose variables and meaning remain represented. Exact label packing supplies no such whole-state abstraction.

Experience records currently retain the exact predicate, prediction, executed assessment and generation context. They do not contain selection probabilities for unchosen actions. Off-policy comparisons therefore require a separately recorded behavior policy, genuine selection propensities, candidate set and an explicit distinction between replayable evaluation and an observed chosen-action outcome.

## Observed consumer

The seven authored questions over equal, opposite, independent, refined and extended relations used the complete original BF16 Qwen instrument. All 16 integration checks passed, including recovery, correction, carrier identity and current experience eligibility. The model obtained 3/7 correct coarse labels and 1/7 correct labels with the intended cause. Its wrong predictions remain preserved. This is a measured baseline for using the representation, rather than a learned-inquiry result.

The integration suite passed 77 tests; its six native tests initially required explicit dependency paths and then all six passed in a separate run. The added extension invariant check also passed within a 13-test focused run. The resulting 84 distinct passing tests cover the current source. The [results](results.md) retain exact receipts, resources and earlier training limits. General relation compression, useful autonomous question formation and joint-core learning retain their own evaluation requirements.

## Task-relative completion and continuing inquiry

When a task asks a question over represented joint alternatives, one query can be settled (`state: "decided"`, `verdict` in `{"yes", "no"}`) while other queries on the same `finite_relation_v1` state remain open (`state: "both"`, `verdict: "maybe"`, retaining witnesses for both `0` and `1`).

- `agent.inquire` (and CLI `inquire`) presents the observed `finite_relation_v1` state and selected prior observations to the core, validates the proposed `relation.query`, `relation.refine`, `relation.extend` or `relation.map` operation through `Steps`, and returns `next_state_step_id` plus an observation-attached snapshot for subsequent work.
- `Scratchboard.relation` (CLI `board-note`) and `documents.relation_note` (CLI `note`) render a source-cited deliverable from a `confirmed` `relation.query` whose enclosure is `decided`, while binding optional `open_step_ids` (`state: "both"` queries on the same task state with `0` and `1` witnesses) as explicit retained open questions. Unreviewed, open (`both`) or conflicting (`conflict`) queries cannot be delivered as settled answers.
- `Memory.completed` and `Memory.status` project `open_questions` per task and record `completed_question`, `evaluator_identity`, `state_operator`, `state_scope` and `uncertainty` on the delivered relational artifact outcome. Conditional `relation.refine` and conservative `relation.extend` preserve their distinct transition scopes across continuing inquiry.

