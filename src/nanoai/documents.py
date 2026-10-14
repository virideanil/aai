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


VISUAL_RULES = (
    {
        "id": "visual.hierarchy",
        "name": "Monotonic Heading Hierarchy",
        "contract": "Start with a single H1 title and advance heading levels monotonically without skipping levels.",
    },
    {
        "id": "visual.table_alignment",
        "name": "Uniform Table Geometry",
        "contract": "Every Markdown table requires a header row, a delimiter row and identical column counts across all rows.",
    },
    {
        "id": "visual.code_fences",
        "name": "Closed Language-Tagged Code Fences",
        "contract": "Every fenced code block must specify a language tag on its opening fence and close cleanly.",
    },
    {
        "id": "visual.citation_grounding",
        "name": "Verified Source Attribution",
        "contract": "Every bracketed citation [source@rev] must match an admitted source reference.",
    },
    {
        "id": "visual.perception_grounding",
        "name": "SEE Region Coordinate Fidelity",
        "contract": "Referenced visual region IDs and bounding boxes [x, y, w, h] must match measured SEE perception records.",
    },
    {
        "id": "visual.density_balance",
        "name": "Scannable Structural Density",
        "contract": "Sections must contain non-empty body content and balanced inline formatting.",
    },
)


def audit_visual_presentation(
    content: str,
    *,
    allowed_sources=None,
    perception_regions=None,
    require_title: bool = True,
):
    """Deterministically audit Markdown/visual output against nano'ai's visual presentation rules."""
    import re
    from .memory import document_references

    if not isinstance(content, str):
        raise ValueError("Visual audit content must be a string")

    lines = content.splitlines()
    violations = []
    in_fence = False
    fence_open_line = None
    code_block_count = 0
    headings = []
    table_blocks = []
    current_table = []

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_open_line = idx
                code_block_count += 1
                lang = stripped[3:].strip()
                if not lang:
                    violations.append({
                        "rule_id": "visual.code_fences",
                        "line": idx,
                        "message": "Opening code fence is missing an explicit language tag",
                    })
            else:
                in_fence = False
                fence_open_line = None
            continue

        if in_fence:
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            current_table.append((idx, stripped))
        else:
            if current_table:
                table_blocks.append(current_table)
                current_table = []

        m_head = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m_head:
            level = len(m_head.group(1))
            title_text = m_head.group(2).strip()
            headings.append((idx, level, title_text))
            if not title_text:
                violations.append({
                    "rule_id": "visual.density_balance",
                    "line": idx,
                    "message": "Heading has empty title text",
                })

    if current_table:
        table_blocks.append(current_table)

    if in_fence:
        violations.append({
            "rule_id": "visual.code_fences",
            "line": fence_open_line or len(lines),
            "message": "Unclosed fenced code block",
        })

    # Rule 1: Heading hierarchy
    if require_title:
        h1_list = [h for h in headings if h[1] == 1]
        if not headings or headings[0][1] != 1:
            violations.append({
                "rule_id": "visual.hierarchy",
                "line": headings[0][0] if headings else 1,
                "message": "Document must begin with a top-level H1 (#) heading",
            })
        elif len(h1_list) > 1:
            violations.append({
                "rule_id": "visual.hierarchy",
                "line": h1_list[1][0],
                "message": "Document contains multiple H1 headings; use H2 (##) or deeper for sections",
            })
    prev_level = 0
    for idx, level, _ in headings:
        if prev_level > 0 and level > prev_level + 1:
            violations.append({
                "rule_id": "visual.hierarchy",
                "line": idx,
                "message": f"Heading level jumped from H{prev_level} to H{level}",
            })
        prev_level = level

    # Rule 2: Table geometry
    for block in table_blocks:
        if len(block) < 2:
            violations.append({
                "rule_id": "visual.table_alignment",
                "line": block[0][0],
                "message": "Markdown table must have at least a header row and a delimiter row",
            })
            continue
        col_counts = [len([c for c in row_str.strip("|").split("|")]) for _, row_str in block]
        delim_cells = [c.strip() for c in block[1][1].strip("|").split("|")]
        if not all(re.match(r"^:?-{3,}:?$", c) for c in delim_cells):
            violations.append({
                "rule_id": "visual.table_alignment",
                "line": block[1][0],
                "message": "Second row of Markdown table must be a valid delimiter row (|---|)",
            })
        if len(set(col_counts)) != 1:
            violations.append({
                "rule_id": "visual.table_alignment",
                "line": block[0][0],
                "message": f"Inconsistent table column counts across rows: {col_counts}",
            })

    # Rule 4: Citation grounding
    refs = sorted(document_references(content))
    if allowed_sources is not None:
        allowed_set = set(allowed_sources)
        for ref in refs:
            if ref not in allowed_set:
                violations.append({
                    "rule_id": "visual.citation_grounding",
                    "line": 1,
                    "message": f"Citation [{ref}] is not in allowed sources {sorted(allowed_set)}",
                })

    # Rule 5: SEE perception grounding
    perception_mentions = re.findall(r"\[region:(\d+)\s+bbox=\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]", content)
    if perception_regions is not None:
        expected_regions = {
            int(r["region_id"]): tuple(int(v) for v in r["bbox"])
            for r in perception_regions
            if "region_id" in r and "bbox" in r
        }
        for reg_str, x_s, y_s, w_s, h_s in perception_mentions:
            rid = int(reg_str)
            bbox = (int(x_s), int(y_s), int(w_s), int(h_s))
            if rid not in expected_regions:
                violations.append({
                    "rule_id": "visual.perception_grounding",
                    "line": 1,
                    "message": f"Referenced SEE region:{rid} is not present in measured perception records",
                })
            elif expected_regions[rid] != bbox:
                violations.append({
                    "rule_id": "visual.perception_grounding",
                    "line": 1,
                    "message": f"Referenced SEE region:{rid} bbox {list(bbox)} differs from measured {list(expected_regions[rid])}",
                })

    # Rule 6: Density & non-empty content
    if not content.strip():
        violations.append({
            "rule_id": "visual.density_balance",
            "line": 1,
            "message": "Document content is empty",
        })

    failed_ids = {v["rule_id"] for v in violations}
    rules_checked = [
        {
            "id": rule["id"],
            "name": rule["name"],
            "status": "failed" if rule["id"] in failed_ids else "passed",
        }
        for rule in VISUAL_RULES
    ]
    passed_count = sum(1 for r in rules_checked if r["status"] == "passed")
    return {
        "schema": "nanoai.visual-audit.v1",
        "passed": len(violations) == 0,
        "score": round(passed_count / len(VISUAL_RULES), 6),
        "rules_checked": rules_checked,
        "violations": violations,
        "metrics": {
            "heading_count": len(headings),
            "table_count": len(table_blocks),
            "code_block_count": code_block_count,
            "citation_count": len(refs),
            "citations": refs,
            "perception_ref_count": len(perception_mentions),
        },
    }


def render_visual_report(
    title: str,
    sections: list[dict],
    *,
    citations=(),
    perception_regions=(),
) -> str:
    """Deterministically render a structured Markdown visual report guaranteed to satisfy VISUAL_RULES."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Visual report requires a non-empty title")
    if not isinstance(sections, (list, tuple)) or not 1 <= len(sections) <= 32:
        raise ValueError("Visual report requires between 1 and 32 sections")
    lines = [f"# {title.strip()}"]
    if citations:
        cite_str = " ".join(f"[{c}]" for c in citations)
        lines.extend(["", f"Grounded sources: {cite_str}"])
    for sec in sections:
        if not isinstance(sec, dict) or not isinstance(sec.get("heading"), str) or not sec["heading"].strip():
            raise ValueError("Each visual report section requires a non-empty heading")
        lines.extend(["", f"## {sec['heading'].strip()}"])
        if sec.get("body"):
            lines.extend(["", str(sec["body"]).strip()])
        if sec.get("table"):
            tbl = sec["table"]
            headers = [str(h) for h in tbl.get("headers", ())]
            rows = tbl.get("rows", ())
            if not headers or any(len(r) != len(headers) for r in rows):
                raise ValueError("Visual report table requires headers and rows of matching width")
            lines.append("")
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for r in rows:
                lines.append("| " + " | ".join(str(cell) for cell in r) + " |")
        if sec.get("code"):
            code_spec = sec["code"]
            lang = str(code_spec.get("language", "text")).strip() or "text"
            snippet = str(code_spec.get("snippet", "")).rstrip("\n")
            lines.extend(["", f"```{lang}", snippet, "```"])
    if perception_regions:
        lines.extend(["", "## Visual Perception Grounding", ""])
        lines.append("| Region ID | Bounding Box | Reference Tag |")
        lines.append("| --- | --- | --- |")
        for reg in perception_regions:
            rid = int(reg["region_id"])
            bx = [int(v) for v in reg["bbox"]]
            tag = f"[region:{rid} bbox=[{bx[0]}, {bx[1]}, {bx[2]}, {bx[3]}]]"
            lines.append(f"| `{rid}` | `{bx}` | `{tag}` |")
    rendered = "\n".join(lines) + "\n"
    audit = audit_visual_presentation(
        rendered,
        allowed_sources=citations if citations else None,
        perception_regions=perception_regions if perception_regions else None,
    )
    if not audit["passed"]:
        raise ValueError(f"Rendered visual report failed visual rules: {audit['violations']}")
    return rendered


