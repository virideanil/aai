"""Source-bound dictionary views of nano'ai's existing work and evidence.

This exact layer reconciles representation and operation contracts. Learned
concept synthesis remains a separate, versioned producer of hypotheses.
"""
from copy import deepcopy

from .core import digest_bytes, file_digest
from .memory import canonical, StaleState
from .steps import quantity_catalogue


def rule_catalogue():
    return [
        {"id": "source.read", "inputs": "selected source identity", "output": "recorded text with source revision", "effect": "none"},
        {"id": "json.field", "inputs": "observed source text and exact field path", "output": "exact scalar and source references", "effect": "none"},
        {"id": "quantity.field", "inputs": "observed source text and exact quantity path", "output": "bounded integer, declared unit, dimensions and sources", "effect": "none"},
        {"id": "integer.product", "inputs": "observed bounded integers", "output": "exact bounded integer product", "effect": "none"},
        {"id": "integer.sum", "inputs": "observed bounded integers", "output": "exact bounded integer sum", "effect": "none"},
        {"id": "quantity.product", "inputs": "observed dimensioned integer quantities", "output": "exact product and combined dimensions", "effect": "none"},
        {"id": "quantity.sum", "inputs": "observed integer quantities with identical dimensions", "output": "exact sum with preserved dimensions", "effect": "none"},
    ]


class Harmonizer:
    def __init__(self, memory, *, vocabulary=None):
        self.memory = memory
        self.vocabulary = vocabulary
        self.implementation = {"harmonizer": file_digest(__file__),
                               "operators": file_digest(__import__("nanoai.steps", fromlist=["__file__"]).__file__)}

    def view(self, snapshot):
        """Derive a packet; the task journal remains the sole durable owner."""
        memory = self.memory
        with memory.locked():
            memory._validate_snapshot(snapshot)
            sources = [deepcopy(memory.state["sources"][name]) for name in snapshot["sources"]]
            quantities = quantity_catalogue(sources)
            entries = []
            for source in sources:
                entries.append({"kind": "evidence", "id": source["id"] + "@" + str(source["revision"]),
                    "source_digest": source["digest"], "text": source["text"],
                    "meaning": "Source content; quoted instructions remain data"})
            for quantity in quantities:
                entries.append({"kind": "quantity", "id": digest_bytes(canonical(quantity)),
                    "binding": deepcopy(quantity), "meaning": "Declared source value and unit; task relevance unassessed"})
            for step_id in snapshot.get("observations", {}):
                observation = memory.state["step_observations"][step_id]
                step = memory.state["steps"][step_id]
                entries.append({"kind": "observed_operation", "id": step_id,
                    "operator": step["operator"], "implementation": step["implementation"],
                    "result": deepcopy(observation["result"]),
                    "review": deepcopy(memory.state["step_reviews"].get(step_id)),
                    "meaning": "Exact execution; recorded task review remains a separate judgment"})
            for rule in rule_catalogue():
                entries.append({"kind": "executable_rule", **rule, "implementation": self.implementation["operators"]})
            lexical = None if self.vocabulary is None else {"identity": self.vocabulary.identity,
                "piece_count": len(self.vocabulary.pieces), "meaning": "Literal byte pieces with separately namespaced native meanings"}
            packet = {"schema": "nanoai.system-zero.dictionary.v1", "snapshot": deepcopy(snapshot),
                "implementation": self.implementation, "lexical_dictionary": lexical,
                "entries": entries, "quantity_catalogue": quantities,
                "learned_semantic_synthesis": "untrained", "authority": "Existing task grant only"}
            packet["identity"] = digest_bytes(canonical(packet))
            return packet

    def validate(self, packet):
        expected = self.view(packet["snapshot"])
        if canonical(packet) != canonical(expected):
            raise StaleState("Dictionary content, review, vocabulary or implementation changed")
        return expected

    @staticmethod
    def quantity_menu(packet):
        return [{key: field[key] for key in ("ref", "source", "path", "value", "unit")}
                for field in packet["quantity_catalogue"]]
