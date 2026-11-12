"""Seeded meaning-preserving perturbations for the GSM8KPerturbed benchmark.

Noise transforms are adapted from HELM's invariance perturbations
(stanford-crfm/helm, Apache-2.0, src/helm/benchmark/augmentations/) and are
byte-identical to the originals under a shared random.Random. The homophone
swap follows NL-Augmenter's close_homophones_swap (MIT) using exact CMUdict
pronunciation matches. History conditions prepend seeded, unrelated off-domain chat
exchanges (OpenAssistant/oasst2) as prior chat turns.

Only the question text is perturbed; templates, answer-format instructions,
and anything the extractor keys on are never touched.

Design: marin-community/marin#7776 (part of marin-community/marin#7090).
"""

import dataclasses
import functools
import json
import re
from pathlib import Path
from random import Random

_MISSPELLINGS_FILE = Path(__file__).resolve().parent / "correct_to_misspelling.json"
with open(_MISSPELLINGS_FILE) as f:
    CORRECT_TO_MISSPELLING = json.load(f)
_MISSPELL_PATTERN = re.compile(r"\b({})\b".format("|".join(CORRECT_TO_MISSPELLING.keys())))

# Keyboard-adjacency map, verbatim from HELM TyposPerturbation.
KEY_APPROX = {
    "q": "was",
    "w": "qesad",
    "e": "wsdfr",
    "r": "edfgt",
    "t": "rfghy",
    "y": "tghju",
    "u": "yhjki",
    "i": "ujklo",
    "o": "iklp",
    "p": "ol",
    "a": "qwsz",
    "s": "weadzx",
    "d": "erfcxs",
    "f": "rtgvcd",
    "g": "tyhbvf",
    "h": "yujnbg",
    "j": "uikmnh",
    "k": "iolmj",
    "l": "opk",
    "z": "asx",
    "x": "sdcz",
    "c": "dfvx",
    "v": "fgbc",
    "b": "ghnv",
    "n": "hjmb",
    "m": "jkn",
}

# Verbatim from HELM contraction_expansion_perturbation.py (Apache-2.0).
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what're": "what are",
    "what's": "what is",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

REVERSE_CONTRACTION_MAP = {v: k for k, v in CONTRACTION_MAP.items()}
# HELM's exact pattern: trailing-space guard (never contract at end of
# sentence), case-insensitive.
_REVERSE_CONTRACTION_PATTERN = re.compile(
    r"\b({})\b ".format("|".join(REVERSE_CONTRACTION_MAP.keys())),
    flags=re.IGNORECASE | re.DOTALL,
)


def _match_case(source: str, target: str) -> str:
    """Mirror of helm.common.general.match_case."""
    if all(letter.islower() for letter in source):
        return target.lower()
    if all(letter.isupper() for letter in source):
        return target.upper()
    if source and source[0].isupper():
        return target.capitalize()
    return target


def typos(text: str, rng: Random, prob: float = 0.05) -> str:
    out = []
    for letter in text:
        lc = letter.lower()
        if lc in KEY_APPROX and rng.random() < prob:
            new = rng.choice(list(KEY_APPROX[lc]))
        else:
            new = lc
        out.append(new.upper() if lc != letter else new)
    return "".join(out)


def misspellings(text: str, rng: Random, prob: float) -> str:
    def sub(m: re.Match) -> str:
        word = m.group(1)
        if rng.random() < prob:
            return _match_case(word, str(rng.choice(CORRECT_TO_MISSPELLING[word])))
        return word

    return _MISSPELL_PATTERN.sub(sub, text)


def lowercase(text: str, _rng: Random) -> str:
    """`_rng` is unused; the parameter exists for the NOISE_CONDITIONS signature."""
    return text.lower()


def extra_space(text: str, rng: Random, max_spaces: int = 3) -> str:
    return re.sub(r" +", lambda _: " " * rng.randint(1, max_spaces), text)


def contraction(text: str, _rng: Random) -> str:
    def cont(m: re.Match) -> str:
        match = m.group(1)
        contracted = REVERSE_CONTRACTION_MAP.get(match, REVERSE_CONTRACTION_MAP.get(match.lower()))
        return _match_case(match, contracted) + " "

    return _REVERSE_CONTRACTION_PATTERN.sub(cont, text)


def mild_mix(text: str, rng: Random) -> str:
    """HELM's canonical composite: lowercase, contract, misspell, respace."""
    text = lowercase(text, rng)
    text = contraction(text, rng)
    text = misspellings(text, rng, prob=0.1)
    text = extra_space(text, rng, max_spaces=3)
    return text


# Number-word swaps (two->too, for->four) are deliberately allowed in the
# homophone condition: they are the realistic ASR case and recoverable from
# context, unlike digit corruption. NUMBER_WORDS only flags affected items.
NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "half",
    "quarter",
    "won",
    "too",
    "to",
    "for",
    "fore",
    "ate",
}
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


@functools.cache
def _homophone_map() -> dict[str, list[str]]:
    """word -> [exact homophones], built by inverting CMUdict pronunciations."""
    import cmudict  # noqa: PLC0415  (optional dependency, loaded on first use)

    by_pron: dict[tuple, set[str]] = {}
    for word, pron in cmudict.entries():
        if word.isalpha():
            by_pron.setdefault(tuple(pron), set()).add(word)
    homophone_map: dict[str, set[str]] = {}
    for words in by_pron.values():
        if len(words) > 1:
            for w in words:
                homophone_map.setdefault(w, set()).update(words - {w})
    return {w: sorted(cands) for w, cands in homophone_map.items()}


def homophones(text: str, rng: Random, prob: float = 0.1) -> str:
    def sub(m: re.Match) -> str:
        word = m.group(0)
        if len(word) < 2 or rng.random() >= prob:
            return word
        candidates = _homophone_map().get(word.lower(), [])
        if not candidates:
            return word
        return _match_case(word, rng.choice(candidates))

    return _WORD_RE.sub(sub, text)


def number_words_affected(original: str, perturbed: str) -> bool:
    def number_word_multiset(t: str) -> list:
        return sorted(w for w in _WORD_RE.findall(t.lower()) if w in NUMBER_WORDS)

    return number_word_multiset(original) != number_word_multiset(perturbed)


NOISE_CONDITIONS = {
    "misspellings": lambda t, r: misspellings(t, r, prob=0.2),
    "lowercase": lowercase,
    "extra_space": extra_space,
    "typos": typos,
    "homophones": homophones,
    "mild_mix": mild_mix,
}
HISTORY_CONDITIONS = {"history_4": 4, "history_16": 16}
CONDITIONS = list(NOISE_CONDITIONS) + list(HISTORY_CONDITIONS)


def assign_conditions(n_items: int, seed: int, perturbations_per_item: int = 1) -> list[list[str]]:
    """Seeded stratified assignment: each item gets `perturbations_per_item`
    distinct conditions; each condition covers ~n_items/len(CONDITIONS) items
    per round."""
    per_item = min(perturbations_per_item, len(CONDITIONS))
    order = list(range(n_items))
    Random(seed).shuffle(order)
    assigned = [[] for _ in range(n_items)]
    for round_idx in range(per_item):
        for pos, item_idx in enumerate(order):
            assigned[item_idx].append(CONDITIONS[(pos + round_idx) % len(CONDITIONS)])
    return assigned


@dataclasses.dataclass(frozen=True)
class PerturbedQuestion:
    """A perturbed question plus the metadata scoring slices on.

    `changed` lets scoring separate real perturbations from no-ops (which
    otherwise dilute per-condition deltas); `number_words_affected` marks
    recoverable spoken-number swaps for slicing.
    """

    text: str
    changed: bool
    number_words_affected: bool


def perturb_question(question: str, condition: str, seed: int) -> PerturbedQuestion:
    """Apply a noise condition; returns the perturbed text plus slice metadata."""
    text = NOISE_CONDITIONS[condition](question, Random(seed))
    return PerturbedQuestion(
        text=text,
        changed=text != question,
        number_words_affected=number_words_affected(question, text),
    )


# Prime multiplier decorrelating per-item history RNG streams from the noise
# seeds; load-bearing for reproducing the checked-in data.
_HISTORY_SEED_PRIME = 100003


def history_indices(n_pool: int, item_index: int, n_exchanges: int, seed: int) -> list[int]:
    """Seeded history-pool picks; longer histories extend shorter ones."""
    rng = Random(seed * _HISTORY_SEED_PRIME + item_index)
    return rng.sample(range(n_pool), max(HISTORY_CONDITIONS.values()))[:n_exchanges]
