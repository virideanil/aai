"""Freeze authored reading worlds, independent outcomes and family splits."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanoai.core import digest_bytes
from nanoai.memory import canonical


# An entire domain, including its values and question, belongs to one split.
DOMAINS = [
    ("train", "gallery", "en", "Which room hosts the gallery tour?", "The gallery tour is in {}.", ["the east hall", "the garden room", "the upper studio"]),
    ("train", "parcel", "en", "Where should the parcel be collected?", "The parcel collection point is {}.", ["the front desk", "the side entrance", "the depot"]),
    ("train", "workshop", "en", "When does the workshop begin?", "The workshop begins at {}.", ["nine", "eleven", "two"]),
    ("train", "meal", "en", "What is the meal for the visit?", "The meal for the visit is {}.", ["vegetable soup", "rice and beans", "lentil stew"]),
    ("train", "kit", "tr", "Etkinlik için hangi malzeme gerekli?", "Etkinlik için {} gerekli.", ["kâğıt", "ahşap", "kumaş"]),
    ("train", "reading", "tr", "Okuma hangi salonda yapılacak?", "Okuma {} yapılacak.", ["mavi salonda", "yeşil salonda", "küçük odada"]),
    ("validation", "train", "en", "Which platform does the train use?", "The train uses {}.", ["platform one", "platform four", "platform seven"]),
    ("validation", "garden", "tr", "Bahçe hangi gün açılıyor?", "Bahçe {} açılıyor.", ["pazartesi", "çarşamba", "cumartesi"]),
    ("test", "library", "en", "How long may the library book be borrowed?", "The library loan lasts {}.", ["one week", "two weeks", "three weeks"]),
    ("test", "rehearsal", "en", "Who will lead the rehearsal?", "The rehearsal leader is {}.", ["Mira", "Jonas", "Selin"]),
    ("test", "ferry", "tr", "Vapur nereden hareket edecek?", "Vapur {} hareket edecek.", ["kuzey iskelesinden", "eski limandan", "ada iskelesinden"]),
    ("test", "exhibition", "tr", "Sergide hangi eser gösterilecek?", "Sergide {} gösterilecek.", ["deniz resmi", "taş heykel", "cam vazo"]),
]


def build():
    episodes = []
    for split, family, language, question, form, values in DOMAINS:
        tr = language == "tr"
        answer = [form.format(v) for v in values]
        unknown = "Bilgi eksik; güncel bilgiyi sor." if tr else "The information is missing; ask for the current detail."
        conflict = "Kaynaklar çelişiyor; hangi bilginin geçerli olduğunu sor." if tr else "The sources conflict; ask which account is current."
        extra = "Başka bir konuda: pencereler temizlendi." if tr else "On an unrelated matter, the windows were cleaned."
        missing = "Bu konu hakkında henüz bilgi verilmedi." if tr else "No information about this topic has been supplied yet."
        for mode in ("first", "second", "third", "correction", "distraction", "conflict", "missing", "correction_back"):
            base = [answer[0]]
            target, states = 0, ["yes", "no", "no", "no", "no"]
            if mode in ("second", "third"):
                target = 1 if mode == "second" else 2
                base = [answer[target]]; states = ["no"]*5; states[target] = "yes"
            elif mode == "correction":
                base += [("Düzeltme, önceki bilgi geçersiz: " if tr else "Correction, the earlier detail is superseded: ")+answer[1]]
                target, states = 1, ["no", "yes", "no", "no", "no"]
            elif mode == "distraction":
                base += [extra, extra]
            elif mode == "conflict":
                base += [("Aynı tarihten ikinci kaynak şöyle diyor: " if tr else "A second source from the same date says: ")+answer[1]]
                target, states = 4, ["maybe", "maybe", "no", "no", "yes"]
            elif mode == "missing":
                base = [missing, extra]
                target, states = 3, ["maybe", "maybe", "maybe", "yes", "no"]
            elif mode == "correction_back":
                base = [answer[1], extra, ("Son düzeltme: " if tr else "Final correction: ")+answer[0], extra]
            candidates = [{"id": f"option-{i}", "kind": "answer" if i < 3 else "inquire",
                           "content": text} for i, text in enumerate(answer+[unknown, conflict])]
            rng = random.Random(f"nanoai-develop-v1:{family}:{mode}")
            rng.shuffle(candidates)
            verdicts = {f"option-{i}": state for i, state in enumerate(states)}
            episodes.append({"id": family+"-"+mode, "family": family, "split": split,
                "language": language, "mode": mode, "question": question,
                "observations": base, "candidates": candidates,
                "rubric": {"target_id": f"option-{target}", "verdicts": verdicts,
                           "scope": "Authored notice world and exact editable-document choice"}})
    return {"schema": "nanoai.developmental-corpus.v1", "episodes": episodes,
            "source": "Authored in this repository before model feature collection or training",
            "split_unit": "Whole domain including question, values and every trajectory",
            "precision": "Original BF16 backbone, original F32 outputs, new F32 candidate"}


def freeze(path):
    value = build()
    value["identity"] = digest_bytes(canonical(value))
    path = Path(path)
    raw = json.dumps(value, indent=2, ensure_ascii=False)+"\n"
    if path.exists() and path.read_text() != raw:
        raise ValueError("The frozen corpus already has different bytes")
    path.write_text(raw)
    return value


if __name__ == "__main__":
    value = freeze(sys.argv[1])
    texts = {t for e in value["episodes"] for t in [e["question"], *e["observations"], *[c["content"] for c in e["candidates"]]]}
    print(json.dumps({"identity": value["identity"], "episodes": len(value["episodes"]), "unique_texts": len(texts)}))
