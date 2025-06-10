# SPDX-License-Identifier: Apache-2.0
"""
This is based entirely on the ngram spec, with some modifications to test the Blazedit approach instead of the n-gram approach directly. 

This docstring details important information on the testing methodology.

Most of the tests rely on "greedy equality", where we expect the output of
speculative decoding on a sequence to exactly match the output of normal non-
speculative decoding.

Since speculative decoding with rejection sampling guarantees that the output
distribution matches the target model's output distribution (up to hardware
numerics, see https://arxiv.org/pdf/2302.01318.pdf), we can expect greedy
equality.

For ngram lookup, its idea comes from https://github.com/apoorvumang/prompt-lookup-decoding,
and is merged into transform code base: https://github.com/huggingface/transformers/pull/27775.
Since there is no model is needed for generate the proposal, we could make
the testcase much simpler than drafter multi-step one.

However, we still need to verify below scenario could be passed:
    * Batch size 1 greedy equality
    * Batch size >1 greedy equality
    * Test greedy equality under preemption
    * Test greedy equality under various ngram sizes / speculative sizes

With those tests, we can say at least, ngram spec would not break the correctess
for the target model outputs.
"""

import pytest

from ..utils import maybe_enable_chunked_prefill
from .conftest import run_equality_correctness_test


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [
        {
            # Skip cuda graph recording for fast test.
            "enforce_eager": True,
            # Print spec metrics.
            "disable_log_stats": False,
        }
    ],
)
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            "model_name": "JackFram/llama-160m",
        },
    ],
)
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        {
            "speculative_config": {
                "method": "blazedit",
                "num_blazedit_ngram_speculative_tokens": 9,
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 3,
                "prompt_lookup_max": 3,
                "disable_mqa_scorer": False,
            },
        },
        {
            "speculative_config": {
                "method": "blazedit",
                "num_blazedit_ngram_speculative_tokens": 9,
                "model": "JackFram/llama-68m",
                "num_speculative_tokens": 3,
                "prompt_lookup_max": 3,
                "disable_mqa_scorer": True,
            },
        },
    ],
)
@pytest.mark.parametrize(
    "output_len",
    [
        256,
    ],
)
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("prefill_chunk_size", [-1, 4])
@pytest.mark.parametrize("seed", [1])
def test_ngram_e2e_greedy_correctness(
    vllm_runner,
    common_llm_kwargs,
    per_test_common_llm_kwargs,
    baseline_llm_kwargs,
    test_llm_kwargs,
    batch_size: int,
    output_len: int,
    prefill_chunk_size: int,
    seed: int,
):
    """Verify greedy equality on a tiny model with different batch size."""
    maybe_enable_chunked_prefill(prefill_chunk_size, common_llm_kwargs)
    run_equality_correctness_test(
        vllm_runner,
        common_llm_kwargs,
        per_test_common_llm_kwargs,
        baseline_llm_kwargs,
        test_llm_kwargs,
        batch_size,
        max_output_len=output_len,
        seed=seed,
        temperature=0.0,
    )