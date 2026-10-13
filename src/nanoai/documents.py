"""Editable calculation notes rendered from reviewed, versioned observations.

The learned actor selects the operation. This renderer preserves exact values,
units and source references without asking a model to spell their identifiers.
"""
from copy import deepcopy

from .core import file_digest


def quantity_text(result):
    units = " ".join(key + (f"^{power}" if power != 1 else "")
                     for key, power in sorted(result["dimensions"].items()))
    return str(result["value"]) + (" " + units if units else "")


def quantity_note(memory, step_id, *, language="en", filename=None):
    if language not in {"en", "tr"}:
        raise ValueError("Calculation-note language")
    with memory.locked():
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
        task = memory.state["tasks"][snapshot["task_id"]]
        if filename is None:
            if len(task["filenames"]) != 1:
                raise ValueError("Choose one of the task's document filenames")
            filename = task["filenames"][0]
        inputs = [memory.state["step_observations"][sid]["result"] for sid in step["arguments"]["steps"]]
        result = observation["result"]
        refs = result["sources"]
        symbol = " × " if step["operator"] == "quantity.product" else " + "
        expression = symbol.join(quantity_text(value) for value in inputs)
        heading, total = ("Calculation", "Calculated total") if language == "en" else ("Hesaplama", "Hesaplanan toplam")
        citations = " ".join("[" + ref + "]" for ref in refs)
        content = f"# {heading}\n\n{total}: **{quantity_text(result)}**. {citations}\n\n{expression} = {quantity_text(result)}.\n"
    return memory.record_proposal(snapshot, {"filename": filename, "content": content, "citations": refs},
        metrics={"renderer": "typed-calculation-note-v1", "renderer_identity": file_digest(__file__), "render_model_calls": 0})
