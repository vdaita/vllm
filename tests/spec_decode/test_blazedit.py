# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import MagicMock

import pytest
import torch

from vllm.attention.selector import (_Backend,
                                     global_force_attn_backend_context_manager)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import (ExecuteModelRequest, HiddenStates, Logprob,
                           get_all_seq_ids)
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker import Worker
from vllm.spec_decode.blazedit_worker import BlazeditWorker

from .utils import (assert_logprobs_dict_allclose, create_batch,
                    create_seq_group_metadata_from_prompts, create_worker,
                    patch_execute_model_with_seeds, zero_kv_cache)

@torch.inference_mode()
def test_use_draft_model_runner_advance_step():
    """Verify that draft model runner triggers advance step
    when applicable.
    """
    seed = 100
    model_name = 'JackFram/llama-68m'

    k = 5
    batch_size = 32
    block_size = 32
    num_gpu_blocks = 2048 // block_size
    worker = create_worker(
        MultiStepWorker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
        model_runner_cls=TP1DraftModelRunner,
    )

    # Mock "_gpu_advance_step" to raise an exception when called.
    exception_secret = "artificial stop"
    worker.model_runner._gpu_advance_step = MagicMock()
    worker.model_runner._gpu_advance_step.side_effect = ValueError(
        exception_secret)

    seq_group_metadata_list, _, _ = create_batch(batch_size,
                                                 k,
                                                 block_size=block_size,
                                                 num_gpu_blocks=num_gpu_blocks)

    # Fallback (should not call) when num_steps=1.
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        num_lookahead_slots=k,
        num_steps=1)
    worker.execute_model(execute_model_req=execute_model_req)

    # Expect exception if _gpu_advance_step is called.
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        num_lookahead_slots=k,
        num_steps=k)

    with pytest.raises(ValueError, match=exception_secret):
        worker.execute_model(execute_model_req=execute_model_req)
    call_args_list = worker.model_runner._gpu_advance_step.call_args_list
    assert len(call_args_list) == 1