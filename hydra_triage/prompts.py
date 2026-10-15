"""Prompt text for the synthetic logs.

Every request row carries one sampled prompt string. Templates are rendered from a small grammar
(role, brand preamble, instruction, object, format, constraint, chain-of-thought ask) followed by
content that is mostly opaque tokens (ids, numbers, names). The detector never sees which grammar
choices produced a prompt; it has to recover structure reuse from the text (see fingerprint.py).
"""

from __future__ import annotations

import numpy as np

ROLES = (
    "You are a helpful assistant.",
    "You are an expert tutor.",
    "You are a senior engineer.",
    "You are a careful analyst.",
    "You are a patient teacher.",
    "Act as a domain expert.",
)
INSTRUCTIONS = (
    "Summarize",
    "Rewrite",
    "Translate into Spanish",
    "Classify",
    "Extract the key fields from",
    "Draft a reply to",
    "Explain",
    "Review",
    "Proofread",
    "Convert to a checklist",
)
OBJECTS = (
    "the following support ticket",
    "this customer email",
    "the document below",
    "the following code snippet",
    "this product description",
    "the meeting notes below",
    "the following contract clause",
    "this user review",
    "the text below",
    "the following log excerpt",
    "this job posting",
    "the transcript below",
)
FORMATS = (
    "as JSON",
    "in three bullet points",
    "in one paragraph",
    "as a numbered list",
    "in plain English",
    "as a markdown table",
    "in under 50 words",
    "as a short memo",
)
CONSTRAINTS = (
    "Do not add commentary.",
    "Keep the original tone.",
    "Be concise.",
    "Cite the source line.",
    "Use formal language.",
    "Return only the result.",
    "Preserve all names.",
    "Flag anything uncertain.",
)
# Chain-of-thought elicitation phrasings. Rings draw a disjoint slice of three each (see synth.py).
COT_ASKS = (
    "Think step by step and show all of your reasoning before the final answer.",
    "Work through the problem step by step, writing out every intermediate step.",
    "Reason carefully, showing each step, then give the final answer on its own line.",
    "Explain your reasoning in full detail before you answer.",
    "Show your complete chain of thought, then the answer.",
    "Walk through the solution one step at a time and justify each step.",
    "Lay out all intermediate reasoning, then conclude.",
    "Before answering, write out the reasoning that leads to the answer.",
    "Solve this carefully and show your work.",
    "Give a detailed derivation followed by the final result.",
    "Think aloud through each stage of the problem.",
    "Provide full worked reasoning and then the result.",
)
PRODUCT_WORDS = (
    "billing",
    "scheduling",
    "inventory",
    "recruiting",
    "analytics",
    "support",
    "compliance",
    "logistics",
    "payroll",
    "marketing",
    "research",
    "procurement",
    "travel",
    "legal",
    "security",
    "education",
)
# Filler words that may appear inside content and survive rare-token masking (realistic noise).
FILLERS = (
    "please",
    "urgent",
    "customer",
    "order",
    "invoice",
    "account",
    "error",
    "thanks",
    "update",
    "status",
    "request",
    "payment",
    "shipping",
    "refund",
    "login",
    "report",
)
SYNONYMS = {
    "step": ("stage", "part"),
    "reasoning": ("thinking", "logic", "rationale"),
    "final": ("ultimate", "last", "concluding"),
    "answer": ("result", "solution", "response"),
    "show": ("display", "present", "write out"),
    "carefully": ("thoroughly", "rigorously", "closely"),
    "problem": ("question", "task", "exercise"),
    "following": ("next", "given", "attached"),
    "explain": ("describe", "clarify", "spell out"),
    "detail": ("depth", "specifics"),
    "complete": ("entire", "full", "whole"),
    "before": ("prior to", "ahead of"),
    "every": ("each", "all the"),
    "then": ("afterwards", "next", "finally"),
    "give": ("provide", "state", "return"),
    "work": ("working", "steps"),
    "solve": ("work out", "resolve"),
    "write": ("set", "put"),
}


def brand_word(rng: np.random.Generator) -> str:
    """A pseudo-word product/brand name. Rare across the corpus unless many accounts share it."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    n = int(rng.integers(5, 9))
    return "".join(letters[i] for i in rng.integers(0, 26, size=n)).capitalize()


def render_template(
    rng: np.random.Generator,
    *,
    role: bool,
    brand: str | None,
    cot: str | None = None,
    fmt: bool = True,
    constraint: bool = True,
    product: bool = False,
) -> str:
    """One templated instruction (no content). Each call draws fresh grammar choices."""
    parts: list[str] = []
    if role:
        parts.append(str(rng.choice(ROLES)))
    if brand is not None:
        if product:
            w1, w2 = rng.choice(PRODUCT_WORDS, size=2, replace=False)
            parts.append(f"You are the assistant for {brand}, a {w1} and {w2} tool.")
        else:
            parts.append(f"You are the assistant for {brand}.")
    sent = f"{rng.choice(INSTRUCTIONS)} {rng.choice(OBJECTS)}"
    if fmt:
        sent += f" {rng.choice(FORMATS)}"
    parts.append(sent + ".")
    if constraint:
        parts.append(str(rng.choice(CONSTRAINTS)))
    if cot is not None:
        parts.append(cot)
    return " ".join(parts)


def free_form(rng: np.random.Generator, n: int, cot: bool = False) -> list[str]:
    """n independent free-form instructions (no role, no brand): the shape of ad-hoc human prompts."""
    i = rng.integers(0, len(INSTRUCTIONS), size=n)
    o = rng.integers(0, len(OBJECTS), size=n)
    f = rng.integers(0, len(FORMATS) + 1, size=n)  # == len → no format clause
    c = rng.integers(0, len(CONSTRAINTS) + 1, size=n)
    k = rng.integers(0, len(COT_ASKS), size=n)
    out = []
    for a, b, d, e, g in zip(i, o, f, c, k):
        s = f"{INSTRUCTIONS[a]} {OBJECTS[b]}"
        if d < len(FORMATS):
            s += f" {FORMATS[d]}"
        s += "."
        if e < len(CONSTRAINTS):
            s += f" {CONSTRAINTS[e]}"
        if cot:
            s += f" {COT_ASKS[g]}"
        out.append(s)
    return out


def fresh_structures(rng: np.random.Generator, n: int, cot: bool) -> list[str]:
    """n prompts each with its own structure: random role/brand/grammar/CoT phrasing per prompt.
    This is what an attacker who rewrites every prompt into a new form produces."""
    role = rng.integers(0, len(ROLES) + 1, size=n)  # == len → no role
    brand = rng.random(n) < 0.5
    brands = (
        np.ascontiguousarray(np.array(list("abcdefghijklmnopqrstuvwxyz"))[rng.integers(0, 26, size=(n, 6))])
        .view("<U6")
        .ravel()
    )
    i = rng.integers(0, len(INSTRUCTIONS), size=n)
    o = rng.integers(0, len(OBJECTS), size=n)
    f = rng.integers(0, len(FORMATS) + 1, size=n)
    c = rng.integers(0, len(CONSTRAINTS) + 1, size=n)
    k = rng.integers(0, len(COT_ASKS), size=n)
    out = []
    for j in range(n):
        parts = []
        if role[j] < len(ROLES):
            parts.append(ROLES[role[j]])
        if brand[j]:
            parts.append(f"You are the assistant for {brands[j].capitalize()}.")
        sent = f"{INSTRUCTIONS[i[j]]} {OBJECTS[o[j]]}"
        if f[j] < len(FORMATS):
            sent += f" {FORMATS[f[j]]}"
        parts.append(sent + ".")
        if c[j] < len(CONSTRAINTS):
            parts.append(CONSTRAINTS[c[j]])
        if cot:
            parts.append(COT_ASKS[k[j]])
        out.append(" ".join(parts))
    return out


def content(rng: np.random.Generator, n: int, filler_rate: float = 0.12) -> list[str]:
    """n content payloads of 6-14 tokens: mostly opaque ids/pseudo-words, a few filler words."""
    lengths = rng.integers(6, 15, size=n)
    total = int(lengths.sum())
    ids = rng.integers(10**6, 10**9, size=total).astype(str)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    words = np.ascontiguousarray(letters[rng.integers(0, 26, size=(total, 6))]).view("<U6").ravel()
    kind = rng.random(total)
    fill = rng.choice(np.array(FILLERS), size=total)
    toks = np.where(kind < filler_rate, fill, np.where(kind < 0.5, ids, words))
    joined = " ".join(toks)  # one join, then split on precomputed character offsets
    tok_len = np.char.str_len(toks) + 1
    ends = np.cumsum(tok_len)
    row_end = ends[np.cumsum(lengths) - 1] - 1
    row_start = np.concatenate([[0], row_end[:-1] + 1])
    return [joined[a:b] for a, b in zip(row_start, row_end)]


def paraphrase(text: str, rng: np.random.Generator, p: float) -> str:
    """Swap a fraction p of swappable words for synonyms. Light paraphrase keeps most shingles."""
    toks = text.split(" ")
    for i, t in enumerate(toks):
        key = t.lower().strip(".,:")
        if key in SYNONYMS and rng.random() < p:
            rep = str(rng.choice(SYNONYMS[key]))
            if t[:1].isupper():
                rep = rep.capitalize()
            toks[i] = rep + t[len(key) :] if t.lower().startswith(key) else rep
    return " ".join(toks)
