"""Exact Krea2 Qwen3-VL conditioning/token alignment and two subject roles.

Template stripping follows ComfyUI and FreeFuse; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
import unicodedata

KREA2_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def _normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold())


@dataclass(frozen=True)
class PromptInfo:
    prompt: str
    token_ids: tuple[int, ...]
    surface: str
    char_positions: tuple[int, ...]
    _conditioning: object = field(default=None, repr=False, compare=False)

    def matches(self, positive):
        return self._conditioning is not None and self._conditioning is positive


@dataclass(frozen=True)
class Subject:
    id: str
    prompt_info: PromptInfo
    phrase: str
    positions: tuple[int, ...]
    manual_mask: object = field(default=None, repr=False, compare=False)


def build_info(prompt, token_row, decode) -> PromptInfo:
    ids = []
    for pair in token_row:
        if (len(pair) < 2 or isinstance(pair[0], bool) or not isinstance(pair[0], int)
                or float(pair[1]) != 1.):
            raise ValueError("Krea2 alignment requires ordinary integer tokens with weight 1")
        ids.append(pair[0])
    starts = [i for i, token_id in enumerate(ids) if token_id == 151644]
    if len(starts) not in (2, 3):
        raise ValueError("Krea2 template is missing or repeated; expected one system/user template")
    end = starts[1]
    if ids[end + 1:end + 3] != [872, 198]:
        raise ValueError("Krea2 template user prefix differs from the supported encoder")
    ids = ids[end + 3:]
    if not ids:
        raise ValueError("Krea2 prompt has no tokens after template stripping")
    surface = ""; positions = []
    for index, token_id in enumerate(ids):
        # Decode full prefixes to handle byte-level tokens rather than treating
        # each individual token's replacement character as actual prompt text.
        current = _normalize(str(decode(ids[:index + 1])))
        common = 0
        while common < min(len(surface), len(current)) and surface[common] == current[common]:
            common += 1
        positions = positions[:common] + [index] * (len(current) - common)
        surface = current
    return PromptInfo(str(prompt), tuple(ids), surface, tuple(positions))


def resolve_phrase(info: PromptInfo, phrase: str, occurrence=0) -> tuple[int, ...]:
    phrase_normal = _normalize(str(phrase)).strip()
    if not phrase_normal or isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0:
        raise ValueError("Provide a nonempty subject phrase and nonnegative occurrence")
    matches = []
    for match in re.finditer(re.escape(phrase_normal), info.surface):
        start, end = match.span()
        if ((start and phrase_normal[0].isalnum() and info.surface[start - 1].isalnum())
                or (end < len(info.surface) and phrase_normal[-1].isalnum() and info.surface[end].isalnum())):
            continue
        matches.append(tuple(dict.fromkeys(info.char_positions[start:end])))
    if occurrence >= len(matches):
        raise ValueError(f"Subject phrase {phrase!r} has {len(matches)} token matches; occurrence {occurrence} is unavailable")
    return matches[occurrence]


def make_subjects(info, target_phrase, protected_phrase, target_occurrence=0, protected_occurrence=0,
                  target_mask=None, protected_mask=None):
    target = resolve_phrase(info, target_phrase, target_occurrence)
    protected = resolve_phrase(info, protected_phrase, protected_occurrence)
    if set(target) & set(protected):
        raise ValueError("Target/protected phrase token positions overlap; choose distinct person descriptions")
    if (target_mask is None) != (protected_mask is None):
        raise ValueError("Provide both manual masks or neither; mixed manual/auto is unsupported")
    return (Subject("target", info, target_phrase, target, target_mask),
            Subject("protected", info, protected_phrase, protected, protected_mask))


def encode_prompt(clip, prompt):
    if not isinstance(prompt, str) or not prompt.strip() or "<|im_start|>" in prompt:
        raise ValueError("Enter an ordinary full-scene prompt, without chat template markers")
    # The native Krea2 tokenizer applies its own template exactly once.
    tokens = clip.tokenize(prompt)
    try:
        rows = tokens["qwen3vl_4b"]
        branch = clip.tokenizer.qwen3vl_4b
        tokenizer = getattr(branch, "tokenizer", branch)
        decode = tokenizer.decode
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("Use CLIPLoader type krea2 with its Qwen3-VL-4B encoder") from error
    if len(rows) != 1:
        raise ValueError("Krea2 Slider FreeFuse supports one token row without prompt scheduling")
    info = build_info(prompt, rows[0], decode)
    positive = clip.encode_from_tokens_scheduled(tokens)
    if len(positive) != 1:
        raise ValueError("Prompt scheduling/multiple conditioning entries are unsupported")
    value, metadata = positive[0]
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 30720:
        raise ValueError("Expected Krea2 conditioning with shape [1, tokens, 30720]")
    if value.shape[1] != len(info.token_ids):
        raise ValueError("Encoded Krea2 token count differs from the template-stripped token positions")
    if set(metadata) - {"pooled_output", "attention_mask"}:
        raise ValueError("Custom conditioning metadata is unsupported")
    return positive, replace(info, _conditioning=positive)
