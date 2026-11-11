import os
import tempfile

# Unit tests never reach the Hugging Face Hub, and never read this machine's own Hub cache: set before
# huggingface_hub is imported (it reads both at import).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = tempfile.mkdtemp(prefix="aris-server-tests-hub-")

import mlx.core as mx  # noqa: E402
import pytest  # noqa: E402

# Unit tests never need the GPU.
mx.set_default_device(mx.cpu)


@pytest.fixture(autouse=True)
def _forget_hub_answers():
    """The Hub's answers are kept for the life of a process; each test asks afresh."""
    from aris_server.engine import hub_info

    forget = getattr(hub_info, "cache_clear", lambda: None)
    forget()
    yield
    forget()
