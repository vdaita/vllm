# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import MagicMock

import pytest
import torch

from vllm.attention.selector import _Backend, global_force_attn_backend_context_manager
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import ExecuteModelRequest, HiddenStates, Logprob, get_all_seq_ids
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.spec_decode.blazedit_worker import BlazeditWorker
from vllm.worker.worker import Worker
from vllm.spec_decode.ngram_worker import NGramWorker


from .utils import (
    assert_logprobs_dict_allclose,
    create_batch,
    create_seq_group_metadata_from_prompts,
    create_worker,
    patch_execute_model_with_seeds,
    zero_kv_cache,
)


@torch.inference_mode()
def test_multi_step_with_batch_expansion_correct_output():
    """
    In this test we verify that the MultiStepWorker is able to handle bonus
    tokens correctly. The test verifies that if a sequence has a
    bonus token then the MultiStepWorker is able to expand the batch by adding
    new sequences corresponding to the sequences with bonus tokens. The
    expanded batch is then used for predicting the next tokens.
    """
    seed = 100
    model_name = "JackFram/llama-68m"

    block_size = 16
    num_gpu_blocks = 2048 // block_size
    batch_size = 128

    ngram_worker = create_worker(
        NGramWorker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )
    ngram_worker.set_ngram_window_size(1, 3)

    multi_step_worker = create_worker(
        BlazeditWorker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
        model_runner_cls=TP1DraftModelRunner,
        model_kwargs={
            "ngram_worker": ngram_worker,
            "num_ngram_steps": 2
        }
    )
    multi_step_worker.set_include_gpu_probs_tensor()

    worker = create_worker(
        Worker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )
    random.seed(seed)
    prompts = [[0] for _ in range(batch_size)]
    num_steps = 2
    final_prompt_lens = [(num_steps + 1) for prompt in prompts]
    rand_seeds = list(random.randint(0, 100) for _ in range(num_steps))
    multi_step_worker.execute_model = patch_execute_model_with_seeds(
        multi_step_worker, rand_seeds
    )
    worker.execute_model = patch_execute_model_with_seeds(worker, rand_seeds)
    # Create the test continuations
    continuations = [[random.randint(0, 1000)] for _ in prompts]
    seq_group_metadata_list = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        continuations=continuations,
        final_prompt_lens=final_prompt_lens,
    )

    # Run single-step twice to generate 2 tokens. This
    # will simulate the bonus token case with the second token
    # being the bonus token.
    zero_kv_cache(worker.cache_engine)

    single_step_output: list[SamplerOutput] = []
    set_random_seed(seed)
    for _ in range(num_steps):
        seq_group_metadata_list = create_seq_group_metadata_from_prompts(
            prompts,
            num_gpu_blocks,
            block_size,
            continuations=continuations,
            final_prompt_lens=final_prompt_lens,
        )
        single_step_output.extend(
            worker.execute_model(
                execute_model_req=ExecuteModelRequest(
                    seq_group_metadata_list=seq_group_metadata_list
                )
            )
        )
        # Append output tokens to new sequence data.
        for i, seq_group_output in enumerate(single_step_output[-1]):
            continuations[i].append(seq_group_output.samples[0].output_token)

    # Create continuations for the MultiStepWorker. The continuations have
    # 2 tokens in order to simulate the bonus token case.
    multi_step_continuations = []
    for continuation in continuations:
        multi_step_continuations.append(continuation[:2])
    seq_group_metadata_list = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        continuations=multi_step_continuations,
        final_prompt_lens=final_prompt_lens,
    )

    # Run multi-step and verify that the third token prediction is accurate
    # for all sequences.
    zero_kv_cache(multi_step_worker.worker.scorer_worker.cache_engine)


    all_seq_ids = {i for i in range(batch_size)}
    multi_step_output, _ = multi_step_worker.sampler_output(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            num_lookahead_slots=1
        ),
        seq_ids_with_bonus_token_in_last_step=all_seq_ids,
    )
    for index, output in enumerate(multi_step_output[-1].outputs):
        assert continuations[index][-1] == output.samples[0].output_token