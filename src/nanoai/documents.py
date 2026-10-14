"""Editable calculation notes rendered from reviewed, versioned observations.

The learned actor selects the operation. This renderer preserves exact values,
units and source references without asking a model to spell their identifiers.
"""
from copy import deepcopy

def quantity_text(result):
    units = " ".join(key + (f"^{power}" if power != 1 else "")
                     for key, power in sorted(result["dimensions"].items()))
    return str(result["value"]) + (" " + units if units else "")


def _quantity_components(memory, step_id, language):
    """Render the existing contract while the caller holds the memory lock."""
    if language not in {"en", "tr"}:
        raise ValueError("Calculation-note language")
    step = memory.state["steps"][step_id]
    if step["operator"] not in {"quantity.product", "quantity.sum"}:
        raise ValueError("A calculation note needs a dimensioned operation")
    review = memory.state["step_reviews"].get(step_id)
    if not review or review["decision"] != "confirmed":
        raise ValueError("The chosen operation needs an explicit task review before document preparation")
    observation = memory.state["step_observations"][step_id]
    snapshot = deepcopy(step["snapshot"])
    snapshot.setdefault("observations", {})[step_id] = observation["digest"]
    memory._validate_snapshot(snapshot)
    inputs = [memory.state["step_observations"][sid]["result"] for sid in step["arguments"]["steps"]]
    result = observation["result"]
    refs = result["sources"]
    symbol = " × " if step["operator"] == "quantity.product" else " + "
    expression = symbol.join(quantity_text(value) for value in inputs)
    heading, total = ("Calculation", "Calculated total") if language == "en" else ("Hesaplama", "Hesaplanan toplam")
    citations = " ".join("[" + ref + "]" for ref in refs)
    content = f"# {heading}\n\n{total}: **{quantity_text(result)}**. {citations}\n\n{expression} = {quantity_text(result)}.\n"
    return snapshot, content, refs


def quantity_note(memory, step_id, *, language="en", filename=None):
    from .scratchboard import Scratchboard
    board = Scratchboard(memory)
    node = board.quantity(step_id, language=language)
    return board.propose_document(node["id"], filename=filename)


def expression_text(expression):
    import json
    return json.dumps(expression, ensure_ascii=False, sort_keys=True)


def _witness_assignment(variables, row):
    import json
    return json.dumps(dict(zip(variables, row)), ensure_ascii=False, sort_keys=True)


def _relation_components(memory, step_id, open_step_ids, language):
    """Render a settled relational decision while preserving open questions on the state."""
    from .memory import StaleState
    if language not in {"en", "tr"}:
        raise ValueError("Relation-note language")
    if (not isinstance(open_step_ids, (list, tuple)) or len(open_step_ids) > 16
            or any(not isinstance(oid, str) for oid in open_step_ids)
            or len(set(open_step_ids)) != len(open_step_ids)
            or step_id in open_step_ids):
        raise ValueError("Open relation questions must be distinct step identities")
    step = memory.state["steps"].get(step_id)
    if not step or step["operator"] != "relation.query":
        raise ValueError("A relation note needs a relation.query operation")
    review = memory.state["step_reviews"].get(step_id)
    if not review or review["decision"] != "confirmed":
        raise ValueError("The chosen relation query needs an explicit task review before document preparation")
    observation = memory.state["step_observations"].get(step_id)
    if not observation:
        raise ValueError("The chosen relation query has no observed result")
    result = observation["result"]
    answer = result.get("value")
    if (not isinstance(answer, dict) or answer.get("type") != "relation_answer_v1"
            or answer.get("state") != "decided" or answer.get("verdict") not in {"yes", "no"}
            or len(answer.get("possible_values", ())) != 1):
        raise ValueError("Task-relative completion requires a decided relation query; open or conflicting questions remain in the continuing state")
    state_step_id = step["arguments"]["step"]
    state_step = memory.state["steps"].get(state_step_id)
    state_obs = memory.state["step_observations"].get(state_step_id)
    if not state_step or not state_obs:
        raise ValueError("Relation query state dependency is unavailable")
    snapshot = deepcopy(step["snapshot"])
    snapshot.setdefault("observations", {})[step_id] = observation["digest"]
    open_answers = []
    for oid in open_step_ids:
        ostep = memory.state["steps"].get(oid)
        oobs = memory.state["step_observations"].get(oid)
        oreview = memory.state["step_reviews"].get(oid)
        if not ostep or ostep["operator"] != "relation.query" or not oobs:
            raise ValueError("Retained open question must be an observed relation.query step")
        if oreview and oreview["decision"] in {"rejected", "cancelled"}:
            raise StaleState("Retained open question was rejected or cancelled")
        if (ostep["snapshot"]["task_id"] != step["snapshot"]["task_id"]
                or ostep["snapshot"]["task_revision"] != step["snapshot"]["task_revision"]
                or ostep["arguments"]["step"] != state_step_id):
            raise ValueError("Retained open question must query the same task relation state")
        oanswer = oobs["result"].get("value")
        if (not isinstance(oanswer, dict) or oanswer.get("type") != "relation_answer_v1"
                or oanswer.get("state") != "both" or oanswer.get("possible_values") != [0, 1]
                or set(oanswer.get("witnesses", {})) != {"0", "1"}):
            raise ValueError("Retained open question must have unresolved both state with 0 and 1 witnesses")
        snapshot["observations"][oid] = oobs["digest"]
        open_answers.append(oanswer)
    memory._validate_snapshot(snapshot)
    refs = result["sources"]
    citations = " ".join("[" + ref + "]" for ref in refs)
    bit = str(answer["possible_values"][0])
    witness = _witness_assignment(answer["variables"], answer["witnesses"][bit])
    question = expression_text(answer["question"])
    state_operator = state_step["operator"]
    state_scope = state_obs["result"]["scope"]
    if language == "en":
        lines = [
            "# Relational Decision",
            "",
            f"Settled question `{question}`: **{answer['verdict']}** (state: `{answer['state']}`, witness {bit}: `{witness}`). {citations}",
            "",
            f"State transition: `{state_operator}` ({state_scope}). Enclosure scope: {result['scope']}. Model scope: {answer['model_scope']}.",
        ]
        if open_answers:
            lines.extend(["", "## Retained Open Questions", ""])
            for oanswer in open_answers:
                oq = expression_text(oanswer["question"])
                w0 = _witness_assignment(oanswer["variables"], oanswer["witnesses"]["0"])
                w1 = _witness_assignment(oanswer["variables"], oanswer["witnesses"]["1"])
                lines.append(f"- `{oq}`: **{oanswer['verdict']}** (state: `{oanswer['state']}`, witness 0: `{w0}`, witness 1: `{w1}`).")
    else:
        lines = [
            "# İlişkisel Karar",
            "",
            f"Kesinleşen soru `{question}`: **{answer['verdict']}** (durum: `{answer['state']}`, tanık {bit}: `{witness}`). {citations}",
            "",
            f"Durum geçişi: `{state_operator}` ({state_scope}). Kapsam: {result['scope']}. Model kapsamı: {answer['model_scope']}.",
        ]
        if open_answers:
            lines.extend(["", "## Açık Kalan Sorular", ""])
            for oanswer in open_answers:
                oq = expression_text(oanswer["question"])
                w0 = _witness_assignment(oanswer["variables"], oanswer["witnesses"]["0"])
                w1 = _witness_assignment(oanswer["variables"], oanswer["witnesses"]["1"])
                lines.append(f"- `{oq}`: **{oanswer['verdict']}** (durum: `{oanswer['state']}`, tanık 0: `{w0}`, tanık 1: `{w1}`).")
    content = "\n".join(lines) + "\n"
    return snapshot, content, refs


def relation_note(memory, step_id, *, open_step_ids=(), language="en", filename=None):
    from .scratchboard import Scratchboard
    board = Scratchboard(memory)
    node = board.relation(step_id, open_step_ids=open_step_ids, language=language)
    return board.propose_document(node["id"], filename=filename)

