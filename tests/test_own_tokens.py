"""A request whose history spells a cached reply in other tokens takes the reply's own (engine.align_tokens,
MovingPromptCache.own_tokens), and token_bytes, which tells whether two tokenizations spell the same bytes."""
import logging
import random

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from aris_server.engine import RESYNC_BYTES, MovingPromptCache, _first_difference, align_tokens, token_bytes

# A toy vocabulary: every string of one to three letters over "ab", and a turn's end.
PIECES = [p for n in (1, 2, 3) for p in __import__("itertools").product("ab", repeat=n)]
PIECES = ["".join(p) for p in PIECES] + ["<E>"]
ID = {p: i for i, p in enumerate(PIECES)}
END = ID["<E>"]


def of(token: int) -> bytes:
    return PIECES[token].encode()


def spell(tokens) -> bytes:
    return b"".join(of(t) for t in tokens)


def tokenize(text: str, rng: random.Random) -> list[int]:
    """One of the many ways to tokenize `text` in the toy vocabulary, picked at random."""
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, min(3, len(text) - i))
        out.append(ID[text[i:i + n]])
        i += n
    return out


def letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ab") for _ in range(n))


def test_two_tokenizations_of_one_reply_line_up_whole_and_spell_the_request_exactly():
    rng = random.Random(1)
    for trial in range(2000):
        history, reply, after = letters(rng, rng.randint(0, 30)), letters(rng, rng.randint(1, 60)), letters(rng, 20)
        before = tokenize(history, rng)
        key = before + tokenize(reply, rng)                       # the turn, as the model generated its reply
        prompt = before + tokenize(reply, rng) + [END] + tokenize(after, rng)  # its next request, re-tokenized
        start = _first_difference(key, prompt, 0, 0)
        aligned, shared, places = align_tokens(key, prompt, start, of)
        assert spell(aligned) == spell(prompt), trial
        assert shared == len(key) and aligned[:len(key)] == key, trial  # the reply ends where the request's turn does
        assert all(key[si] != prompt[sj] for si, sj in places)


def test_where_the_texts_differ_the_entrys_tokens_stop_short_of_it():
    rng = random.Random(2)
    for trial in range(2000):
        text = letters(rng, rng.randint(2, 80))
        at = rng.randrange(len(text))
        other = text[:at] + ("b" if text[at] == "a" else "a") + text[at + 1:]  # one letter changed
        key = tokenize(text, rng)
        prompt = tokenize(other, rng) + [END]
        start = _first_difference(key, prompt, 0, 0)
        aligned, shared, _ = align_tokens(key, prompt, start, of)
        assert spell(aligned) == spell(prompt), trial
        assert aligned[:shared] == key[:shared]
        assert len(spell(key[:shared])) <= at, trial  # nothing of the entry's past where the text changes


def test_a_request_that_ends_inside_the_entry_takes_what_spells_it():
    key = [ID["ab"], ID["ba"], ID["aa"], END]
    prompt = [ID["a"], ID["bb"], ID["a"]]  # "abba": the entry's first two tokens, split the other way
    aligned, shared, places = align_tokens(key, prompt, 0, of)
    assert (aligned, shared, places) == (key[:2], 2, [(0, 0)])


def test_an_entry_that_ends_inside_a_request_token_keeps_only_what_lines_up_before_it():
    key = [ID["a"], ID["b"], ID["a"]]                 # "aba"
    prompt = [ID["ab"], ID["aa"], END]                # "abaa": its "aa" runs past the entry's end
    aligned, shared, _ = align_tokens(key, prompt, 0, of)
    assert spell(aligned) == spell(prompt) and shared == 2 and aligned == key[:2] + prompt[1:]


def test_tokenizations_that_never_meet_again_within_the_bound_are_left_to_the_request():
    n = RESYNC_BYTES  # "aa" tokens against "a" then "aa"s: their boundaries never meet before the end
    key = [ID["aa"]] * n + [END]
    prompt = [ID["a"]] + [ID["aa"]] * (n - 1) + [ID["a"], END]
    aligned, shared, places = align_tokens(key, prompt, 0, of)
    assert (aligned, shared, places) == (prompt, 0, [])


# --- MovingPromptCache.own_tokens -------------------------------------------------------------------------------


class Layer:
    def __init__(self, nbytes=10):
        self.nbytes = nbytes

    def is_trimmable(self):
        return False


SYSTEM = tokenize("ab" * 40, random.Random(3))


def conversation(seed: int):
    """A turn as generated (its reply in one tokenization) and its next request (the reply in another)."""
    rng = random.Random(seed)
    history, reply = SYSTEM + tokenize(letters(rng, 50), rng), letters(rng, 400)
    generated = tokenize(reply, rng)
    retokenized = tokenize(reply, random.Random(seed + 1000))
    assert generated != retokenized
    tool_result = [END] + tokenize(letters(rng, 30), rng) + [END]
    return history + generated, history + retokenized + tool_result, tool_result


def cache_with(*entries, kind="assistant"):
    cache = MovingPromptCache(max_size=10)
    cache.token_bytes = of
    cache.light_min_tokens = 16
    for key in entries:
        cache.insert_cache("m", key, [Layer()], cache_type=kind)
    return cache


def tokenized(prompt, system=len(SYSTEM)):
    return prompt, [prompt[:system], prompt[system:]], ["system", "assistant"], "normal", None


def test_a_request_takes_its_cached_turns_own_tokens_and_continues_the_turn_whole(caplog):
    turn, request, tool_result = conversation(4)
    cache = cache_with(turn)
    with caplog.at_level(logging.INFO):
        aligned, segments, types, _, _ = cache.own_tokens("m", tokenized(request))
    assert aligned == turn + tool_result and spell(aligned) == spell(request)
    assert segments == [aligned[:len(SYSTEM)], aligned[len(SYSTEM):]] and types == ["system", "assistant"]
    assert f"spells a {len(turn)}-token cached entry's text in other tokens" in caplog.text
    assert "shares all of them" in caplog.text
    layers, rest = cache.fetch_nearest_cache("m", aligned)
    assert len(layers) == 1 and rest == tool_result  # the turn handed over whole


def test_each_later_request_lines_up_at_the_same_places_and_logs_only_a_new_one(caplog):
    turn, request, tool_result = conversation(5)
    cache = cache_with(turn)
    first, *_ = cache.own_tokens("m", tokenized(request))
    rng = random.Random(6)
    reply2 = letters(rng, 300)
    turn2 = first + tokenize(reply2, rng)  # the next turn, generated after the aligned prompt
    cache.fetch_nearest_cache("m", first)
    cache.insert_cache("m", turn2, [Layer()], cache_type="assistant")
    # Its request tokenizes the first reply the canonical way again, and the second one differently too.
    tool_result2 = [END] + tokenize(letters(rng, 20), rng) + [END]
    again = request + tokenize(reply2, random.Random(7)) + tool_result2
    with caplog.at_level(logging.DEBUG):
        caplog.clear()
        aligned, *_ = cache.own_tokens("m", tokenized(again))
    assert aligned == turn2 + tool_result2
    assert [r.levelno for r in caplog.records if "spells a" in r.message] == [logging.INFO]  # a new place
    # The same request once more lines up at the same places: logged only at debug level.
    with caplog.at_level(logging.DEBUG):
        caplog.clear()
        assert cache.own_tokens("m", tokenized(again))[0] == turn2 + tool_result2
    assert [r.levelno for r in caplog.records if "spells a" in r.message] == [logging.DEBUG]


def test_a_turn_saved_to_disk_is_lined_up_with_too():
    turn, request, tool_result = conversation(8)
    cache = cache_with(SYSTEM, kind="system")

    class Disk:
        index = {"turn.safetensors": (tuple(turn), "assistant"), "sys.safetensors": (tuple(SYSTEM), "system")}

    cache.disk = Disk()
    assert cache.own_tokens("m", tokenized(request))[0] == turn + tool_result


def test_a_history_that_spells_other_text_keeps_its_own_tokens():
    turn, request, _ = conversation(9)
    changed = list(request)
    at = len(turn) - 40
    changed[at] = ID["b"] if PIECES[changed[at]] != "b" else ID["a"]  # one token's letters changed
    cache = cache_with(turn)
    aligned, *_ = cache.own_tokens("m", tokenized(changed))
    assert spell(aligned) == spell(changed) and len(spell(aligned)) == len(spell(changed))
    shared = _first_difference(aligned, turn, 0, 0)
    assert _first_difference(changed, turn, 0, 0) <= shared < len(turn)  # lined up only to where the text changes


@pytest.mark.parametrize("why", ["no byte table", "an image", "short"])
def test_requests_it_does_not_apply_to_are_left_as_they_are(why):
    turn, request, _ = conversation(10)
    cache = cache_with(turn)
    t = tokenized(request)
    if why == "no byte table":
        cache.token_bytes = None
    elif why == "an image":
        t = t[:4] + (object(),)
    else:
        cache.light_min_tokens = len(request) + 1
    assert cache.own_tokens("m", t) is t


def test_a_segment_ending_inside_what_the_entrys_tokens_spell_leaves_the_request_as_it_is():
    turn, request, _ = conversation(11)
    cache = cache_with(turn)
    t = tokenized(request, system=len(turn) - 50)  # a segment boundary inside the re-tokenized reply
    assert cache.own_tokens("m", t) is t


def test_an_entry_the_request_already_continues_is_left_alone():
    turn, request, tool_result = conversation(12)
    cache = cache_with(turn)
    t = tokenized(turn + tool_result)
    assert cache.own_tokens("m", t) is t


# --- token_bytes ----------------------------------------------------------------------------------------------------

CORPUS = ["def f(x):\n\treturn {'a': x[0], \"b\": 1.5e-3}", "ünïcödé ✓ 中文 🙂 , . 's n't", "checkOk(\"[::-1]\", [0, 1])"]


def byte_level_tokenizer(clean_up=False):
    """A small byte-level BPE tokenizer, as Qwen's and Inkling's are, trained here: no download."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<|im_end|>", "<|im_start|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS * 20, trainer)
    # transformers 5 ignores clean-up for a BPE tokenizer unless told twice; an older one, or a config that says so,
    # decodes " ." as "." and so isn't byte for byte.
    forced = {"clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output": True} if clean_up else {}
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", clean_up_tokenization_spaces=clean_up,
                                   **forced)


def test_a_byte_level_tokenizers_tokens_spell_its_text_byte_for_byte():
    tok = byte_level_tokenizer()
    spell_of = token_bytes(tok)
    assert spell_of is not None
    text = "x = '中文 🙂'<|im_end|>\n" + CORPUS[0]
    ids = tok.encode(text, add_special_tokens=False)
    assert b"".join(spell_of(t) for t in ids) == text.encode()
    # Split a token that holds part of a character: its bytes still say exactly what it holds.
    emoji = [t for t in tok.encode("🙂", add_special_tokens=False)]
    assert b"".join(spell_of(t) for t in emoji) == "🙂".encode()


def test_tokenizers_whose_tokens_cant_be_compared_byte_for_byte_get_none():
    assert token_bytes(byte_level_tokenizer(clean_up=True)) is None  # decoding drops the space in " ."

    meta = Tokenizer(models.BPE(unk_token="[UNK]"))
    meta.pre_tokenizer = pre_tokenizers.Metaspace()  # SentencePiece's "▁" for spaces, not byte-level
    meta.decoder = decoders.Metaspace()
    meta.train_from_iterator(CORPUS * 20, trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"]))
    assert token_bytes(PreTrainedTokenizerFast(tokenizer_object=meta, unk_token="[UNK]")) is None

    class NotATokenizer:
        pass

    assert token_bytes(NotATokenizer()) is None


def test_a_segment_after_what_lines_up_moves_with_the_request_s_own_tokens():
    turn, request, tool_result = conversation(13)
    cache = cache_with(turn)
    tail = len(request) - 3  # a thinking tail mlx-lm keeps as its own segment, in the request's own tokens
    t = (request, [request[:len(SYSTEM)], request[len(SYSTEM):tail], request[tail:]], ["system", "user", "assistant"],
         "normal", None)
    aligned, segments, types, _, _ = cache.own_tokens("m", t)
    assert aligned == turn + tool_result and types == ["system", "user", "assistant"]
    assert segments == [aligned[:len(SYSTEM)], aligned[len(SYSTEM):-3], aligned[-3:]]
