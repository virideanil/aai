"""Small Python interface; native code owns repeated segmentation and EM work."""
import hashlib
import math
from . import _native
from .reference import MasterTokenizer as ReferenceTokenizer, MAX_INPUT


class Tokenizer(ReferenceTokenizer):
    """Immutable byte law with in-process C++ execution.

    The default record is NMT1-compatible. Optional boundaries force byte offsets
    between pieces; minimum selection first minimizes piece count, then maximizes
    the original lexical path probability. Neither policy changes source bytes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        body = self._body()
        object.__setattr__(self, "_handle", _native.create(body + hashlib.shake_256(body).digest(64)))

    def __setattr__(self, name, value):
        raise AttributeError("Tokenizer snapshots are immutable")

    def __delattr__(self, name):
        raise AttributeError("Tokenizer snapshots are immutable")

    def encode(self, raw, *, objective="likelihood", boundaries=()):
        if not isinstance(raw, bytes) or len(raw) > MAX_INPUT:
            raise ValueError("Token input requires bytes within the declared extent")
        if objective not in ("likelihood", "minimum"):
            raise ValueError("Token objective must be likelihood or minimum")
        boundaries = tuple(boundaries)
        if (any(type(b) is not int or not 0 < b < len(raw) for b in boundaries)
                or boundaries != tuple(sorted(set(boundaries)))):
            raise ValueError("Token boundaries require increasing interior integer byte offsets")
        result = _native.encode(self._handle, raw, objective == "minimum", boundaries)
        if objective != "likelihood" or boundaries:
            result.update(objective=objective, boundaries=list(boundaries))
        else:
            result.pop("selected_logp")
        return result

    def expected_counts(self, raw):
        """Native lexical expectation, with the complete finite-string log law."""
        if not isinstance(raw, bytes) or len(raw) > MAX_INPUT:
            raise ValueError("Token input requires bytes within the declared extent")
        return _native.expected_counts(self._handle, raw)

    def fit(self, rows, *, epochs=4, prior=.05):
        """Fit lexical probabilities in a new version, retaining every piece/ID.

        Training rows stay local; model parameters and embeddings are untouched.
        Evaluate the candidate on separate data before replacing a deployment.
        """
        rows = tuple(rows)
        if not rows or len(rows) > 4096 or any(not isinstance(r, bytes) or not r or len(r) > MAX_INPUT for r in rows):
            raise ValueError("Lexical training needs 1..4096 nonempty byte rows")
        if sum(map(len, rows)) > 4_194_304 or type(epochs) is not int or not 1 <= epochs <= 64:
            raise ValueError("Lexical training work budget")
        if not math.isfinite(prior) or prior <= 0:
            raise ValueError("Lexical training prior must be positive")
        current = self
        for _ in range(epochs):
            counts = [prior] * len(self.pieces)
            for raw in rows:
                expected, _ = current.expected_counts(raw)
                for i, value in enumerate(expected):
                    counts[i] += value
            total = math.fsum(counts)
            current = Tokenizer(self.pieces, self.common_ids, [x / total for x in counts],
                                len(rows) / (len(rows) + total), self.vocabulary_identity, self.inventory_identity)
        return current
