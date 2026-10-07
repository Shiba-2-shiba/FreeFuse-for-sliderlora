from dataclasses import replace

import pytest
import torch

from slider_fuse.conditioning import build_info, resolve_phrase, make_subjects, encode_prompt, KREA2_TEMPLATE


def info():
    tokens = [(151644, 1.), (99, 1.), (151644, 1.), (872, 1.), (198, 1.),
              (1, 1.), (2, 1.), (3, 1.), (1, 1.)]
    decoder = {1: "adult woman", 2: " and ", 3: "adult man; "}
    return build_info("adult woman and adult man; adult woman", tokens, lambda ids: "".join(decoder.get(i, "") for i in ids))


def test_template_strip_repeat_phrase_and_word_boundaries():
    value = info()
    assert value.token_ids == (1, 2, 3, 1)
    assert resolve_phrase(value, "adult woman", 0) == (0,)
    assert resolve_phrase(value, "adult woman", 1) == (3,)
    assert resolve_phrase(value, "adult man", 0) == (2,)
    with pytest.raises(ValueError): resolve_phrase(value, "man", 3)
    with pytest.raises(ValueError): resolve_phrase(value, "adul", 0)


def test_subjects_include_protected_without_an_adapter():
    subjects = make_subjects(info(), "adult woman", "adult man", 0, 0)
    assert [s.id for s in subjects] == ["target", "protected"]
    assert [s.positions for s in subjects] == [(0,), (2,)]
    assert all(not hasattr(s, "lora_path") for s in subjects)
    with pytest.raises(ValueError, match="overlap"):
        make_subjects(info(), "adult woman", "adult woman", 0, 0)


def test_invalid_template_and_prompt_scheduling():
    with pytest.raises(ValueError, match="template"):
        build_info("woman", [(1, 1.)], lambda ids: "woman")


class FakeClip:
    def __init__(self):
        self.calls = []
        self.tokenizer = type("Wrapper", (), {"qwen3vl_4b": type("Branch", (), {"tokenizer": self})()})()
    def decode(self, ids):
        return "".join({1: "woman", 2: " and man"}.get(i, "") for i in ids)
    def tokenize(self, text):
        self.calls.append(text)
        return {"qwen3vl_4b": [[(151644, 1.), (99, 1.), (151644, 1.), (872, 1.), (198, 1.), (1, 1.), (2, 1.)]]}
    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 2, 30720), {}]]


def test_encode_once_and_exact_conditioning_identity():
    clip = FakeClip()
    positive, meta = encode_prompt(clip, "woman and man")
    assert meta.matches(positive)
    assert not meta.matches(list(positive))
    assert len(clip.calls) == 1
    assert clip.calls[0] == "woman and man"
    assert meta.token_ids == (1, 2)


def test_encode_rejects_unexpected_runtime_token_count():
    clip = FakeClip()
    clip.encode_from_tokens_scheduled = lambda tokens: [[torch.zeros(1, 3, 30720), {}]]
    with pytest.raises(ValueError, match="token"):
        encode_prompt(clip, "woman and man")
