"""Source-bound lessons and evidence-backed corrections over the existing journal.

Exact quotation checks establish attribution. A task-owned evaluator qualifies
its corrective targets; neither step certifies unrestricted semantic judgment.
"""
from copy import deepcopy
import json

from .core import digest_bytes, file_digest
from .experience import Experience
from .memory import StaleState, bounded_text, canonical, identifier


def validate_lesson(lesson):
    fields = {"schema", "id", "principle", "scope", "reason", "example", "contrast", "next_check"}
    if not isinstance(lesson, dict) or set(lesson) != fields or lesson["schema"] != "nanoai.lesson.v1":
        raise ValueError("Lesson contract differs")
    identifier(lesson["id"])
    if len(lesson["id"]) > 48:
        raise ValueError("Lesson identity extent")
    for name in ("principle", "scope", "reason", "next_check"):
        bounded_text(lesson[name], "lesson " + name, 1024)
    for name in ("example", "contrast"):
        if not isinstance(lesson[name], dict) or set(lesson[name]) != {"context", "attempt", "feedback", "response"}:
            raise ValueError("Worked lesson contract differs")
        for value in lesson[name].values():
            bounded_text(value, "worked lesson", 2048)
    if len(canonical(lesson)) > 8192:
        raise ValueError("Lesson extent")
    return deepcopy(lesson)


def lesson_text(lesson, *, demonstrations=True):
    """A selected rule with its reason; examples stay distinguishable from canon."""
    lesson = validate_lesson(lesson)
    text = (f"Teaching principle: {lesson['principle']}\n"
            f"Applies to: {lesson['scope']}\nReason: {lesson['reason']}")
    if demonstrations:
        for name in ("example", "contrast"):
            row = lesson[name]
            text += (f"\nSeparate {name}: {row['context']}\nAttempt: {row['attempt']}"
                     f"\nFeedback: {row['feedback']}\nResponse: {row['response']}")
    return text + "\nNext check: " + lesson["next_check"]


class Teaching:
    def __init__(self, memory):
        self.memory = memory
        self.implementation = file_digest(__file__)

    def install(self, lesson, *, expected_revision=0):
        lesson = validate_lesson(lesson)
        source_id = "lesson-" + lesson["id"]
        text = canonical(lesson).decode()
        with self.memory.locked():
            prior = self.memory.state["sources"].get(source_id)
            if prior is not None and prior["text"] == text:
                return deepcopy(prior)
        return self.memory.put_source(source_id, text, expected_revision=expected_revision,
            role="observation", note="Authored teaching example; apply its declared scope to the current task.")

    def sources(self, source_ids):
        if (not isinstance(source_ids, (list, tuple)) or len(source_ids) > 8
                or len(set(source_ids)) != len(source_ids)):
            raise ValueError("Selected lesson extent")
        self.memory.reload()
        rows = []
        for source_id in source_ids:
            source = deepcopy(self.memory.state["sources"][source_id])
            if not self.memory.source_current(source):
                raise StaleState("The selected lesson is stale")
            lesson = validate_lesson(json.loads(source["text"]))
            if source_id != "lesson-" + lesson["id"]:
                raise ValueError("Lesson source identity differs")
            rows.append(source)
        return rows

    def feedback(self, experience_id, *, source_id, draft_quote, evidence,
                 diagnosis, repair, reviewer, rule_ids=()):
        """Bind a teacher's proposed repair to the actual work and current evidence."""
        identifier(source_id)
        for name, value in (("draft quote", draft_quote), ("diagnosis", diagnosis),
                            ("repair", repair), ("reviewer", reviewer)):
            bounded_text(value, name, 2048)
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
            raise ValueError("Feedback needs exact source evidence")
        rules = self.sources(rule_ids)
        owner = Experience(self.memory)
        with self.memory.locked():
            row = deepcopy(owner._current(experience_id))
            if row["schema"] != "nanoai.generated-experience.v1":
                raise ValueError("Feedback requires observed generated work")
            receipt = owner._receipt(experience_id)
            if draft_quote not in row["generation"]["text"]:
                raise ValueError("Feedback quote is absent from the actual draft")
            references = []
            for item in evidence:
                if not isinstance(item, dict) or set(item) != {"source_id", "quote"}:
                    raise ValueError("Feedback evidence contract")
                bounded_text(item["quote"], "evidence quote", 2048)
                if item["source_id"] not in row["snapshot"]["sources"]:
                    raise ValueError("Feedback source is outside the observed task")
                source = self.memory.state["sources"][item["source_id"]]
                if item["quote"] not in source["text"]:
                    raise ValueError("Feedback evidence quote differs from its source")
                references.append({**item, "revision": source["revision"], "digest": source["digest"]})
            parents = [deepcopy(self.memory.state["sources"][sid]) for sid in row["snapshot"]["sources"]]
            parents += [s for s in rules if s["id"] not in row["snapshot"]["sources"]]
            if source_id in {s["id"] for s in parents}:
                raise ValueError("Feedback cannot become its own ancestor")
        # Each public boundary revalidates the captured dependencies. A concurrent
        # source change rejects the proposed feedback before its publication.
        basis = self.memory.snapshot(row["snapshot"]["task_id"], parents, row["snapshot"]["core_identity"])
        basis = self.memory.attach_artifacts(basis, [receipt["operation_id"]])
        body = {"schema": "nanoai.feedback.v1", "experience_id": experience_id,
                "draft_quote": draft_quote, "evidence": references, "diagnosis": diagnosis,
                "repair": repair, "reviewer": reviewer, "rules": list(rule_ids),
                "status": "attributed-proposal", "implementation": self.implementation}
        text = canonical(body).decode()
        prior = self.memory.state["sources"].get(source_id)
        if prior is not None:
            if prior["text"] == text and prior.get("derived_from") == basis:
                return deepcopy(prior)
            raise ValueError("Use a new identity for revised feedback")
        return self.memory.put_source(source_id, text, role="observation",
            note="Evidence-backed proposed correction; semantic and outcome checks are separate.", derived_from=basis)

    def training_pair(self, experience_id):
        """Export eligible teaching only; confirmation labels never enter this path."""
        row = Experience(self.memory).export(experience_id)
        if (row["schema"] != "nanoai.generated-experience.v1"
                or row["eligibility"]["status"] != "checked"):
            raise StaleState("Training requires a current checked generated experience")
        if row["metadata"].get("split") not in {"train", "replay"}:
            raise ValueError("Only declared teaching or replay examples supply training targets")
        target = row["assessment"]["corrective_supervision"]["training_target"]
        if target is None:
            raise ValueError("The assessed experience has no corrective teaching target")
        return {"messages": deepcopy(row["messages"]), "target": target,
                "experience_id": experience_id, "basis": deepcopy(row["snapshot"]),
                "assessment_identity": digest_bytes(canonical(row["assessment"]))}
