"""Build original source-grounded lessons with explicit splits and target checks."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import digest_bytes
from nanoai.memory import canonical
from teaching_tasks import QUESTION, qualify


def build():
    rows = []
    def review(identity, split, family, source, draft, final, quote, required=(), forbidden=(), *, criticism=None, quantity=None):
        decision = "keep" if draft == final else "revise"
        criteria = {"kind": "review", "source": source, "draft": draft, "decision": decision,
                    "required": [[v] if isinstance(v, str) else v for v in required], "forbidden": list(forbidden)}
        if quantity: criteria["quantity"] = quantity
        observations = ["SOURCE: " + source, "DRAFT: " + draft]
        if criticism: observations.append("PROPOSED CRITICISM: " + criticism)
        rows.append({"id": identity, "split": split, "family": family, "kind": "review", "question": QUESTION,
                     "observations": observations, "target": json.dumps({"decision": decision, "evidence": quote, "text": final}, ensure_ascii=False),
                     "criteria": criteria})
    review("train-correction-1", "train", "correction",
        "Neri has a red coat and a copper lamp. Correction: her coat is now grey.",
        "Neri wears a red coat and carries a copper lamp.", "Neri wears a grey coat and carries a copper lamp.",
        "her coat is now grey", ["Neri", "grey", "copper", "lamp"], ["red"])
    review("train-correction-2", "train", "correction",
        "Ece has a white bag and a leather notebook. Düzeltme: çantası sarıdır, beyaz değildir.",
        "Ece carries a white bag and a leather notebook.", "Ece carries a yellow bag and a leather notebook.",
        "çantası sarıdır", ["Ece", "yellow", "bag", "leather", "notebook"], ["white"])
    review("validation-correction", "validation", "correction",
        "Pia has a black hat and a silver whistle. The hat was corrected to orange.",
        "Pia wears a black hat and holds a silver whistle.", "Pia wears an orange hat and holds a silver whistle.",
        "The hat was corrected to orange", ["Pia", "orange", "silver", "whistle"], ["black"])
    review("test-correction-en", "confirmation", "correction",
        "An earlier notice placed the rehearsal in the west room at noon. The current notice moves it to the east hall; noon is unchanged.",
        "The rehearsal is in the west room at noon.", "The rehearsal is in the east hall at noon.",
        "moves it to the east hall", ["rehearsal", "east hall", "noon"], ["west"])
    review("test-correction-tr", "confirmation", "correction",
        "Ozan has a violet box with a wooden bird. Düzeltme: kutusu turuncudur. Tahta kuş aynı kalır.",
        "Ozan carries a violet box containing a wooden bird.", "Ozan carries an orange box containing a wooden bird.",
        "kutusu turuncudur", ["Ozan", "orange", "box", "wooden", "bird"], ["violet"])

    review("train-knowledge-1", "train", "knowledge",
        "Ivo sees a broken cup. The cause is unknown to him.",
        "Ivo knew Sera broke the cup out of anger.", "Ivo saw a broken cup but did not know its cause.",
        "The cause is unknown to him", ["Ivo", "cup", ["did not know", "unknown", "wondered"]], ["Sera", "anger"])
    review("train-knowledge-2", "train", "knowledge",
        "Mara watched Pavel close the window.", "Mara saw Pavel close the window.", "Mara saw Pavel close the window.",
        "Mara watched Pavel close the window")
    review("validation-knowledge", "validation", "knowledge",
        "Suri found an empty nest. She has no evidence of why the birds left.",
        "Suri knew the birds had fled a wolf.", "Suri found the nest empty but did not know why the birds left.",
        "She has no evidence of why the birds left", ["Suri", "nest", ["did not know", "unknown", "wondered"]], ["wolf"])
    review("test-knowledge-uncertain", "confirmation", "knowledge",
        "A sensor reports that the greenhouse door is open. Its log does not identify who opened it.",
        "The sensor proves that Tomas opened the greenhouse door.", "The greenhouse door is open, but who opened it is unknown.",
        "Its log does not identify who opened it", ["greenhouse", "open", ["unknown", "not known", "does not identify"]], ["Tomas"])
    review("test-knowledge-observed", "confirmation", "knowledge",
        "Nola watched Amir place the glass on the shelf.", "Nola saw Amir put the glass on the shelf.", "Nola saw Amir put the glass on the shelf.",
        "Nola watched Amir place the glass on the shelf", criticism="Nola cannot know where the glass went.")

    review("train-causality-1", "train", "causality",
        "The hatch is locked. Turning the key unlocks the hatch and exposes a ladder to dry ground.",
        "Ari held the key and the child was safe.", "Ari unlocked the hatch with the key, and the child climbed the ladder to dry ground.",
        "Turning the key unlocks the hatch", ["Ari", ["unlock", "opened"], "key", "ladder", ["dry", "safe"]], ["merely"])
    review("train-causality-2", "train", "causality",
        "The bridge is already lowered and safe to cross.", "Tali crossed the lowered bridge.", "Tali crossed the lowered bridge.",
        "The bridge is already lowered", criticism="Tali must lower the bridge before crossing.")
    review("validation-causality", "validation", "causality",
        "Pulling the red lever lowers the bridge. Touching the stone has no effect.",
        "Bela touched the stone and lowered the bridge.", "Bela pulled the red lever and lowered the bridge.",
        "Pulling the red lever lowers the bridge", ["Bela", "red lever", "bridge"], ["stone"])
    review("test-causality-action", "confirmation", "causality",
        "Only opening the roof vent releases the trapped smoke. The dial measures heat without changing it.",
        "Enid turned the dial and released the smoke.", "Enid opened the roof vent and released the trapped smoke.",
        "opening the roof vent releases the trapped smoke", ["Enid", "vent", "smoke", ["open", "lift"]], ["dial"])
    review("test-causality-state", "confirmation", "causality",
        "The sluice was opened in the previous scene. Water is already draining through it.",
        "Lio watched water drain through the open sluice.", "Lio watched water drain through the open sluice.",
        "Water is already draining through it", criticism="Lio needs to open the sluice to start the draining.")

    review("train-evidence-1", "train", "evidence",
        "Ada left a brass key inside the clock. Mira does not know why Ada left town.",
        "Mira found the brass key inside the clock.", "Mira found the brass key inside the clock.",
        "Ada left a brass key inside the clock", criticism="The key is unjustified because Mira does not know Ada's motive.")
    review("train-evidence-2", "train", "evidence",
        "The sealed parcel has never been opened. Its contents are unknown.",
        "Jo inspected the diamonds inside the sealed parcel.", "Jo looked at the sealed parcel, whose contents were unknown.",
        "Its contents are unknown", ["Jo", "parcel", ["unknown", "wondered", "did not know"]], ["diamonds"])
    review("validation-evidence", "validation", "evidence",
        "The receipt records that a violin arrived on Monday. Why it was sent is unknown.",
        "A violin arrived on Monday.", "A violin arrived on Monday.",
        "a violin arrived on Monday", criticism="Delete the arrival because its purpose is unknown.")
    review("test-evidence-supported", "confirmation", "evidence",
        "The expedition log records a copper coin beneath the arch. Its owner has not been identified.",
        "A copper coin lay beneath the arch.", "A copper coin lay beneath the arch.",
        "a copper coin beneath the arch", criticism="The coin cannot be there unless its owner is known.")
    review("test-evidence-unsupported", "confirmation", "evidence",
        "The survey confirms an echo behind the wall. Nobody has inspected the space behind it.",
        "The survey discovered a hidden library behind the wall.", "The survey detected an echo behind the wall, but the space has not been inspected.",
        "Nobody has inspected the space behind it", ["echo", "wall", ["not been inspected", "uninspected", "unknown"]], ["library"])

    for identity, split, count, period, mode, noun, wrong in [
        ("train-quantity-1", "train", 4, 6, "consecutive", "pulses", 10),
        ("train-quantity-2", "train", 3, 8, "simultaneous", "timers", 24),
        ("validation-quantity", "validation", 5, 7, "consecutive", "signals", 12),
        ("test-quantity-sequence", "confirmation", 6, 9, "consecutive", "strokes", 15),
        ("test-quantity-parallel", "confirmation", 4, 11, "simultaneous", "alarms", 44)]:
        source = (f"There are {count} {mode} {noun}, each lasting {period} seconds. " +
                  ("They occur one after another." if mode == "consecutive" else "They start and finish together."))
        total = count * period if mode == "consecutive" else period
        review(identity, split, "quantity", source, f"The elapsed duration is {wrong} seconds.",
            f"The elapsed duration is {total} seconds.", source.split(". ")[0] + ".", ["seconds"], [],
            quantity={"count": count, "period": period, "mode": mode, "unit": "seconds"})

    for identity, split, question, observation, answer, accepted in [
        ("replay-sum", "replay", "Return only the sum.", "7 + 8", "15", ["15"]),
        ("replay-tr", "replay", "Translate into English; return only the translation.", "Kapı açık.", "The door is open.", ["the door is open", "door is open"]),
        ("replay-detail", "replay", "Copy only the identifier exactly.", "Record identifier: ZX-48.", "ZX-48", ["zx-48"]),
        ("replay-unknown", "replay", "Answer briefly from the source only: who sent the letter?", "The sender is not identified.", "Unknown.", ["unknown", "not identified", "the sender is unknown"]),
        ("control-sum", "control", "Return only the sum.", "19 + 23", "42", ["42"]),
        ("control-tr", "control", "Translate into English; return only the translation.", "Pencere kapalı.", "The window is closed.", ["the window is closed", "window is closed"]),
        ("control-detail", "control", "Copy only the identifier exactly.", "Record identifier: QF-731.", "QF-731", ["qf-731"]),
        ("control-unknown", "control", "Answer briefly from the source only: what caused the delay?", "The cause of the delay is not recorded.", "Unknown.", ["unknown", "not recorded", "the cause is unknown"])]:
        rows.append({"id": identity, "split": split, "family": identity.split("-", 1)[1], "kind": "plain",
            "question": question, "observations": [observation], "target": answer,
            "criteria": {"kind": "plain", "accepted": accepted}})
    rationales = {
        "train-correction-1": "The later correction replaces red with grey. It leaves the copper lamp unchanged. Revise only the conflicting coat color and quote the correction.",
        "train-correction-2": "The Turkish correction says the bag is yellow, not white. The leather notebook is unaffected. Change the bag color and preserve the other stated details.",
        "validation-correction": "The current correction makes the hat orange. Keep the silver whistle and change the earlier black color to orange.",
        "train-knowledge-1": "Ivo observes the broken cup but does not know its cause. The draft invents both Sera and anger. Remove those claims and preserve the observed cup and his uncertainty.",
        "train-knowledge-2": "Mara directly watched Pavel close the window. The draft states that observation without adding a cause. It is supported, so keep its exact wording.",
        "validation-knowledge": "The empty nest is observed, but the reason for departure is unknown. A wolf is unsupported. Preserve the empty nest and uncertainty while removing the invented cause.",
        "train-causality-1": "Holding the key alone cannot complete the rescue. The source supplies unlocking the hatch and climbing its ladder to dry ground. State those connected actions.",
        "train-causality-2": "The bridge is already lowered and safe. Crossing follows from that current state. The criticism repeats a completed prerequisite, so preserve the draft.",
        "validation-causality": "The red lever lowers the bridge; touching the stone has no effect. Replace the ineffective action with pulling the lever while preserving the resulting bridge state.",
        "train-evidence-1": "The source explicitly places the key inside the clock. Mira's lack of knowledge about Ada's motive does not remove that observed object. Reject the criticism and keep the draft.",
        "train-evidence-2": "The parcel has never been opened, so diamonds are an invented observation. Preserve Jo and the sealed parcel, with its contents unknown.",
        "validation-evidence": "The receipt supports the violin arriving on Monday. The unknown purpose does not contradict that arrival. Keep the supported sentence exactly.",
        "train-quantity-1": "Four pulses occur consecutively, so their durations add: 4 times 6 equals 24 seconds. Ten seconds is incorrect. Revise the total and retain the unit.",
        "train-quantity-2": "The three timers start and finish together. Elapsed duration is one timer's eight seconds, not their summed durations. Replace 24 with 8 seconds.",
        "validation-quantity": "Five consecutive seven-second signals take 5 times 7, or 35 seconds. Consecutive intervals add; revise the incorrect total of 12 seconds.",
        "replay-sum": "Seven plus eight equals fifteen. The request asks only for the sum, so give 15.",
        "replay-tr": "Kapı means the door and açık means open. Preserve the declarative meaning in English: the door is open.",
        "replay-detail": "The stated identifier is ZX-48. Copy that exact sequence, including its hyphen, without the surrounding label or punctuation.",
        "replay-unknown": "The source says the sender is not identified. It provides no name. Answer that the sender is unknown.",
    }
    for row in rows:
        if row["split"] in {"train", "validation", "replay"}:
            row["teaching_reason"] = rationales[row["id"]]
            row["reason_provenance"] = "Original authored explanation checked against this row's source and target; no model-generated thought is treated as ground truth."
    corpus = {"schema": "nanoai.teaching-tasks.v2", "provenance": "Original authored small procedural curriculum, separate from literary quality and real-world knowledge.",
              "rows": rows}
    corpus["identity"] = digest_bytes(canonical(corpus))
    return corpus


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "evals/teaching/tasks-v2.json"
    corpus = build(); result = qualify(corpus)
    path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(path), "identity": corpus["identity"], **result}, indent=2))
