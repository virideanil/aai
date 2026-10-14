"""Continuing editable writing on the shared source, experience and effect path.

The caller selects the phase and sources. The learned core supplies its actual
text; phase selection and literary judgment retain their own evaluation.
"""
from copy import deepcopy
import json

from nanoai_tokens import load

from .catalogue import LanguageCatalogue
from .core import file_digest
from .experience import Experience
from .harmonizer import Harmonizer, context_messages, source_presentation
from .memory import bounded_text, identifier


class Writing:
    def __init__(self, memory, tokenizer=None):
        self.memory = memory
        self.tokenizer = tokenizer if tokenizer is not None else load()
        self.implementation = file_digest(__file__)

    def _source(self, name, text, *, note, derived_from=None):
        self.memory.reload()
        prior = self.memory.state["sources"].get(name)
        if (prior is not None and prior["text"] == text and prior["note"] == note
                and prior.get("derived_from") == derived_from):
            return deepcopy(prior)
        return self.memory.put_source(name, text, role="observation", note=note,
            expected_revision=prior["revision"] if prior else 0, derived_from=derived_from)

    def _task(self, name, question, filename):
        self.memory.reload()
        prior = self.memory.state["tasks"].get(name)
        if prior and prior["goal"] == question and prior["filenames"] == [filename]:
            return deepcopy(prior)
        return self.memory.set_task(name, question, filenames=[filename],
                                    expected_revision=prior["revision"] if prior else 0)

    def run(self, phase_id, instruction, source_ids, core, *, workflow="book",
            kind="draft", max_tokens=None, filename="text.md", crash_after_publish=False,
            lesson_ids=(), observation_ids=(), scratchboard_ids=()):
        identifier(phase_id); identifier(workflow)
        if len(phase_id) > 58 or kind not in {"plan", "draft", "critique", "revision", "continuation", "control"}:
            raise ValueError("Writing phase identity or kind")
        bounded_text(instruction, "writing instruction", 2048)
        if max_tokens is not None and (type(max_tokens) is not int or not 1 <= max_tokens <= 262144):
            raise ValueError("Writing generation bound")
        if (not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 12
                or len(source_ids) != len(set(source_ids))
                or phase_id + "-input" in source_ids or phase_id + "-output" in source_ids):
            raise ValueError("Writing sources must be distinct existing parents")
        self.memory.reload()
        sources = [deepcopy(self.memory.state["sources"][name]) for name in source_ids]
        from .teaching import Teaching
        lessons = Teaching(self.memory).sources(lesson_ids)
        if set(source_ids) & set(lesson_ids) or len(sources) + len(lessons) > 14:
            raise ValueError("Writing evidence and lesson sources must have distinct bounded roles")
        sources += lessons
        observations = [source_presentation(source, lesson=source["id"] in lesson_ids)["text"] for source in sources]
        self._task(phase_id, instruction, filename)
        inputs = self._source(phase_id + "-input", json.dumps({"observations": observations}, ensure_ascii=False),
                              note=f"Exact generation inputs for {workflow}/{kind}.")
        snapshot = self.memory.snapshot(phase_id, sources + [inputs], core.identity)
        context = Harmonizer(self.memory).context(snapshot, observation_ids=observation_ids,
                                                 scratchboard_ids=scratchboard_ids)
        snapshot = context["snapshot"]
        messages = context_messages(context, output="artifact", phase=kind,
                                    source_ids=[row["id"] for row in sources], lesson_ids=lesson_ids)
        metadata = {"workflow": workflow, "phase": phase_id, "kind": kind,
                    "writer": self.implementation, "max_tokens": max_tokens,
                    "tokenizer_identity": self.tokenizer.identity,
                    "lessons": list(lesson_ids)}
        owner = Experience(self.memory)
        prior = [row for row in self.memory.state["experiences"].values()
                 if row.get("schema") == "nanoai.generated-experience.v1"
                 and row["snapshot"] == snapshot and row["messages"] == messages
                 and all(row["metadata"].get(key) == value for key, value in metadata.items())
                 and row["implementation"] == owner.implementation]
        if len(prior) > 1:
            raise ValueError("Writing phase has ambiguous retained generations")
        if prior:
            row, model_calls = deepcopy(prior[0]), 0
        else:
            # Validate all input bounds before the costly call or any model effect.
            for text in observations:
                bounded_text(text, "writing observation", 8192)
            for message in messages:
                bounded_text(message["content"], "writing message", 16384)
            generated = core.generate(messages, max_tokens=max_tokens, trace=True)
            metadata["native_generation"] = generated
            row = owner.record_generation(snapshot, question=instruction, observations=observations,
                messages=messages, generation={"text": generated["output"],
                    "token_ids": generated["token_ids"], "finish_reason": generated["finish_reason"],
                    "policy": generated.get("sampling", "greedy-token-v1")}, representation_identity=self.tokenizer.identity,
                source_id=inputs["id"], metadata=metadata)
            model_calls = 1
        receipt = owner.execute(row["id"], filename=filename, crash_after_publish=crash_after_publish)
        output = None
        if row["generation"]["text"].strip():
            basis = self.memory.attach_artifacts(row["snapshot"], [receipt["operation_id"]])
            output = self._source(phase_id + "-output", row["generation"]["text"],
                note=f"Generated {kind} for {workflow}; proposal assessed separately.",
                derived_from=basis)
            LanguageCatalogue(self.memory).observe(f"{phase_id}-tokens-{output['revision']}",
                phase_id, output["id"], self.tokenizer)
        outcome = self.memory.artifact_outcome(receipt["operation_id"])
        return {"phase": phase_id, "experience_id": row["id"], "model_calls": model_calls,
                "generation": deepcopy(row["generation"]), "receipt": receipt,
                "source": output, "complete": outcome["allows_completion"], "outcome": outcome,
                "generation_finished": row["generation"]["finish_reason"] == "stop",
                "artifact_path": str(self.memory.root / receipt["path"])}

    def practice(self, phase_id, instruction, source_ids, core, *, criteria, evaluate,
                 feedback, evaluator_identity, scope, lesson_ids=(), max_attempts=3,
                 max_tokens=None, workflow="book", kind="revision"):
        """Revise after observed task feedback within an explicit attempt budget.

        The caller owns the task rubric and explains failed checks. A passed
        mechanical rubric has its named scope; literary judgment remains separate.
        Every draft and check retains its receipt. This method changes no weights.
        """
        identifier(phase_id)
        if len(phase_id) > 48 or type(max_attempts) is not int or not 1 <= max_attempts <= 4:
            raise ValueError("Writing practice budget")
        attempts, selected = [], None
        current_sources = list(source_ids)
        for index in range(max_attempts):
            name = f"{phase_id}-r{index}"
            row = self.run(name, instruction, current_sources, core, workflow=workflow,
                           kind=kind, max_tokens=max_tokens, lesson_ids=lesson_ids)
            assessment = Experience(self.memory).assess_generation(row["experience_id"],
                rubric={"scope": scope, "criteria": criteria, "training_target": None},
                evaluator_identity=evaluator_identity, evaluate=evaluate)
            row["assessment"] = assessment
            row["outcome"] = self.memory.artifact_outcome(row["receipt"]["operation_id"])
            row["complete"] = row["outcome"]["allows_completion"]
            attempts.append(row)
            if assessment["observed_outcome"] == "yes":
                selected = row
                break
            if index + 1 == max_attempts:
                break
            # Journal encoding sorts mapping keys. Give callbacks the same order
            # before and after recovery so feedback does not invent a new source.
            checks = {key: assessment["checks"][key] for key in sorted(assessment["checks"])}
            explanation = feedback(row["generation"]["text"], checks, deepcopy(criteria))
            bounded_text(explanation, "observed writing feedback", 4096)
            experience = self.memory.state["experiences"][row["experience_id"]]
            basis = self.memory.attach_artifacts(experience["snapshot"], [row["receipt"]["operation_id"]])
            review = self._source(name + "-review", explanation,
                note="Observed task checks with an actionable repair; the declared rubric defines the scope.", derived_from=basis)
            current_sources = list(source_ids) + [review["id"]]
            if row["source"] is not None:
                current_sources.append(row["source"]["id"])
        return {"attempts": attempts, "selected": selected, "satisfied": selected is not None,
                "scope": scope, "parameter_updates": 0, "attempt_budget": max_attempts}

    def quote_history(self, experience_id, source_id):
        """Observe an older output as historical text, retaining its original basis."""
        identifier(source_id)
        owner = Experience(self.memory)
        with self.memory.locked():
            row = deepcopy(self.memory.state["experiences"][experience_id])
            if row.get("schema") != "nanoai.generated-experience.v1":
                raise ValueError("Historical writing needs a generated experience")
            receipt = owner._receipt(experience_id)
            actual = self.memory._artifact_path(receipt["path"]).read_text()
            if actual != owner.content(row):
                raise ValueError("Historical writing differs from its observed output")
        # This observation asserts what was written, independently of whether its
        # old fictional claims or task inputs are current.
        basis = self.memory.snapshot(row["snapshot"]["task_id"], [], "historical-writing-v1")
        basis = self.memory.attach_artifacts(basis, [receipt["operation_id"]])
        text = json.dumps({"kind": "historical-writing",
            "original_sources": {k: v["revision"] for k, v in row["snapshot"]["sources"].items()},
            "original_task_revision": row["snapshot"]["task_revision"],
            "text": row["generation"]["text"]}, ensure_ascii=False)
        return self._source(source_id, text,
            note="Historical draft quoted for revision. Current canon takes precedence.", derived_from=basis)

    def assemble(self, workflow, source_ids, title):
        """Publish a current manuscript from separately editable, observed units."""
        identifier(workflow)
        bounded_text(title, "manuscript title", 256)
        if not source_ids or len(source_ids) > 12 or len(source_ids) != len(set(source_ids)):
            raise ValueError("Manuscript source extent")
        self.memory.reload()
        sources = [deepcopy(self.memory.state["sources"][sid]) for sid in source_ids]
        task_id = workflow + "-manuscript"
        self._task(task_id, f"Assemble the current manuscript: {title}", "manuscript.md")
        snapshot = self.memory.snapshot(task_id, sources, self.implementation)
        content = "# " + title + "\n\n" + "\n\n".join(
            f"{row['text']}\n\n[{row['id']}@{row['revision']}]" for row in sources) + "\n"
        output = {"filename": "manuscript.md", "content": content,
                  "citations": [f"{row['id']}@{row['revision']}" for row in sources]}
        existing = [p for p in self.memory.state["proposals"].values()
                    if p["snapshot"] == snapshot and p["output"] == output]
        proposal = existing[0] if existing else self.memory.record_proposal(snapshot, output,
                metrics={"workflow": workflow, "kind": "deterministic-assembly", "model_calls": 0})
        receipt = self.memory.commit(proposal["id"], core_identity=self.implementation)
        return {"receipt": receipt, "artifact_path": str(self.memory.root / receipt["path"]),
                "units": source_ids, "model_calls": 0}
