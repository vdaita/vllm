# SPDX-License-Identifier: Apache-2.0

import copy
import weakref
from typing import Dict, List, Set, Tuple

import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.sequence import (ExecuteModelRequest, HiddenStates, SequenceData,
                           SequenceGroupMetadata)

if current_platform.is_cuda_alike():
    from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner

from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.worker.worker_base import DelegateWorkerBase
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker
from vllm.model_executor.layers.rejection_sampler import RejectionSampler

class BlazeditWorker(MultiStepWorker):
    """
    Proposer worker for Blazedit inference, based on multi_step_worker.py
    """
    def __init__(self, *args, **kwargs):
        if "ngram_worker" in kwargs and "num_ngram_steps" in kwargs:
            ngram_worker = kwargs.pop("ngram_worker")
            num_ngram_steps = kwargs.pop("num_ngram_steps")
            spec_decode_worker = SpecDecodeWorker(
                proposer_worker=ngram_worker,
                scorer_worker=copy.deepcopy(self.worker),
                spec_decode_sampler=RejectionSampler()
            )

            self.worker = spec_decode_worker
            self.num_ngram_steps = num_ngram_steps # move this before the normal initialization

        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int]
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)
        # Expand the batch for sequences with a bonus token.
        # Perform a forward pass on the expanded batch and filter the
        # response to retain only the original sequences' responses.
        expanded_request, indices_of_seq_with_bonus_tokens =\
            self._expand_execute_model_request(
                execute_model_req, seq_ids_with_bonus_token_in_last_step)
        expanded_request.num_lookahead_slots = self.num_ngram_steps
        # Run model sample_len times.
        model_outputs: List[SamplerOutput] = []
        if expanded_request.previous_hidden_states is not None:
            self.worker.model_runner.return_hidden_states = True

        for _ in range(execute_model_req.num_lookahead_slots):
            # Execute the model with the n-gram worker.
            model_output: List[SamplerOutput] = self.worker.execute_model(
                execute_model_req=expanded_request
            )
            model_output = model_output[0]
            print("Model output: ", model_output)

            self._maybe_update_previous_hidden_states(
                model_output, expanded_request)
            self._append_new_tokens(
                model_output, expanded_request.seq_group_metadata_list,
                indices_of_seq_with_bonus_tokens)
            model_outputs.append(model_output)

        # move indices to device to avoid stream sync
        indices_of_seq_with_bonus_tokens = torch.tensor(
            indices_of_seq_with_bonus_tokens, device=self.device)
        filtered_model_outputs = self._filter_model_output(
            model_outputs, indices_of_seq_with_bonus_tokens)

        print("Model outputs after filtering: ", filtered_model_outputs)

        return filtered_model_outputs, True
