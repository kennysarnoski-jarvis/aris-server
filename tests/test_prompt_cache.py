from aris_server.engine import MovingPromptCache


class Layer:
    def __init__(self, nbytes, trimmable=False):
        self.nbytes, self._trimmable = nbytes, trimmable

    def is_trimmable(self):
        return self._trimmable


SYSTEM = list(range(100))
TURN1 = SYSTEM + list(range(100, 300))


def test_a_conversation_moves_to_its_next_turn_and_the_system_prompt_is_copied():
    cache = MovingPromptCache(max_size=10)
    sys_layers, conv_layers = [Layer(10)], [Layer(50)]
    cache.insert_cache("m", SYSTEM, sys_layers, cache_type="system")
    cache.insert_cache("m", TURN1, conv_layers, cache_type="assistant")
    assert cache.nbytes == 60

    got, rest = cache.fetch_nearest_cache("m", TURN1 + [7, 8])
    assert got is conv_layers and rest == [7, 8]        # handed over, not copied
    assert cache.nbytes == 10 and len(cache) == 1       # no longer counted twice

    got, rest = cache.fetch_nearest_cache("m", SYSTEM + [5])
    assert got is not sys_layers and rest == [5]        # a copy: other conversations still start from it
    assert cache.nbytes == 10 and len(cache) == 1


def test_exact_match_and_no_match():
    cache = MovingPromptCache(max_size=10)
    layers = [Layer(5)]
    cache.insert_cache("m", TURN1, layers, cache_type="user")
    assert cache.fetch_nearest_cache("m", TURN1) == (layers, [])
    assert cache.fetch_nearest_cache("m", TURN1) == (None, TURN1)


def test_only_the_system_prompt_segment_is_snapshotted_mid_turn():
    from aris_server.engine import merge_prompt_tail

    prompt = list(range(10))
    tokenized = (prompt, [prompt[:3], prompt[3:8], prompt[8:]], ["system", "user", "assistant"], "reasoning", None)
    assert merge_prompt_tail(tokenized) == (prompt, [prompt[:3], prompt[3:]], ["system", "assistant"], "reasoning", None)
    plain = (prompt, [prompt], ["assistant"], "normal", None)
    assert merge_prompt_tail(plain) == plain
    no_tail = (prompt, [prompt[:3], prompt[3:]], ["system", "user"], "normal", None)
    assert merge_prompt_tail(no_tail) == no_tail  # the last segment is never snapshotted mid-turn
