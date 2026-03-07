# SPDX-License-Identifier: Apache-2.0

import enum
import time
from typing import TYPE_CHECKING, Optional, Union

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:

    from vllm.lora.request import LoRARequest
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.inputs import PlaceholderRange


class Request:

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: list[int],
        multi_modal_inputs: Optional[list["MultiModalKwargs"]],
        multi_modal_hashes: Optional[list[str]],
        multi_modal_placeholders: Optional[list["PlaceholderRange"]],
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional["LoRARequest"] = None,
        structured_output_request: Optional["StructuredOutputRequest"] = None,
        is_streaming_prompt: bool = False,
        is_streaming_prompt_finished: bool = False,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = structured_output_request

        self.status = (RequestStatus.WAITING_FOR_FSM
                       if sampling_params.guided_decoding is not None else
                       RequestStatus.WAITING)
        self.events: list[EngineCoreEvent] = []
        self.last_event_type: Optional[EngineCoreEventType] = None
        self.last_event_timestamp: Optional[float] = None
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: list[str] = multi_modal_hashes or []

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)

        # Read-only views
        # Prevent directly appending to the these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        self.is_streaming_prompt = is_streaming_prompt
        self.is_streaming_prompt_finished = is_streaming_prompt_finished

        # Timing for streaming requests - tracks when the last chunk arrived
        self.arrival_time = arrival_time
        self.last_chunk_arrival_time: Optional[float] = arrival_time

        # Track cumulative tokens invalidated across all prompt updates
        self.total_tokens_invalidated: int = 0

        # Flag to indicate that cache blocks were invalidated in this step.
        # When True, the scheduler should set resumed_from_preemption=True
        # to force the worker to reset the block table row instead of appending.
        self.cache_invalidated_this_step: bool = False

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
        return cls(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(
                sampling_params=request.sampling_params),
            is_streaming_prompt=request.is_streaming_prompt,
            is_streaming_prompt_finished=request.is_streaming_prompt_finished,
        )

    def append_output_token_ids(
        self,
        token_ids: Union[int, list[int]],
    ) -> None:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)

    def append_streaming_prompt_tokens(self,
                                       request: EngineCoreRequest) -> None:

        assert self.is_streaming_prompt, "This should only be called for streaming prompt requests"
        # if you are streaming prompt tokens, you should not have any output tokens
        assert len(
            self._output_token_ids
        ) == 0, "If request is still streaming prompt tokens, then the scheduler should not have generated output tokens"
        assert not self.is_streaming_prompt_finished, "If request is still streaming prompt tokens, then the scheduler should not have marked the request as finished. There is a either a bug or request was EnginerCoreRequest was received out of order"

        # print("Appending streaming prompt tokens")
        # print(f"Prompt token ids: {request.prompt_token_ids}")
        # self.prompt = self.prompt + request.prompt if isinstance(self.prompt, str) else ""
        self.prompt_token_ids.extend(request.prompt_token_ids)
        self._all_token_ids.extend(request.prompt_token_ids)

        # Update last_chunk_arrival_time on every chunk arrival for stream-based scheduling
        self.last_chunk_arrival_time = time.time()

        if request.is_streaming_prompt_finished:
            self.is_streaming_prompt_finished = True

    def update_prompt_tokens(self,
                             request: EngineCoreRequest) -> tuple[int, int]:
        """Update prompt tokens with a new set of tokens.

        This method finds the longest common prefix between the current and new
        token lists, and updates the prompt from the divergence point onwards.
        It returns information about which portion needs to be recomputed.

        Args:
            request: EngineCoreRequest containing the new prompt tokens

        Returns:
            tuple: (common_prefix_length, tokens_to_invalidate)
                - common_prefix_length: Number of tokens in common prefix
                - tokens_to_invalidate: Number of computed tokens to invalidate
        """
        assert self.is_streaming_prompt, "This should only be called for streaming prompt requests"
        assert len(
            self._output_token_ids
        ) == 0, "Cannot update prompt after output tokens have been generated"

        new_token_ids = request.prompt_token_ids

        # Find the longest common prefix
        common_prefix_len = 0
        for i in range(min(len(self.prompt_token_ids), len(new_token_ids))):
            if self.prompt_token_ids[i] == new_token_ids[i]:
                common_prefix_len += 1
            else:
                break

        # Calculate how many computed tokens need to be invalidated
        # (tokens beyond the common prefix that have been computed)
        tokens_to_invalidate = max(
            0, self.num_computed_tokens - common_prefix_len)

        # Update the prompt token lists from the divergence point
        self.prompt_token_ids = new_token_ids
        self._all_token_ids = new_token_ids.copy()

        # Update num_prompt_tokens to reflect new prompt size
        self.num_prompt_tokens = len(new_token_ids)

        # Adjust num_computed_tokens to not exceed the common prefix
        self.num_computed_tokens = min(self.num_computed_tokens,
                                       common_prefix_len)

        # Update last_chunk_arrival_time for stream-based scheduling
        self.last_chunk_arrival_time = time.time()

        if request.is_streaming_prompt_finished:
            self.is_streaming_prompt_finished = True

        # Accumulate total tokens invalidated across all updates
        self.total_tokens_invalidated += tokens_to_invalidate

        return common_prefix_len, tokens_to_invalidate

    def check_is_streaming_prompt(self) -> bool:
        if self.is_streaming_prompt:
            return self.is_streaming_prompt

    def check_is_streaming_prompt_finished(self) -> bool:
        if self.is_streaming_prompt_finished:
            return self.is_streaming_prompt_finished

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def has_encoder_inputs(self) -> bool:
        return len(self.mm_inputs) > 0

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_positions)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id]["length"]
        return num_tokens

    @property
    def use_structured_output(self) -> bool:
        return self.sampling_params.guided_decoding is not None

    @property
    def is_kv_on_gpu(self) -> bool:
        return self.last_event_type == EngineCoreEventType.KV_ON_GPU

    def record_event(
        self,
        event_type: EngineCoreEventType,
        start_timestamp: Optional[float] = None,
    ) -> None:
        if self.last_event_type is not None:
            self.events.append(
                EngineCoreEvent.new_event(
                    self.last_event_type,
                    self.last_event_timestamp,
                    start_timestamp,
                ))
        self.last_event_timestamp = start_timestamp if start_timestamp is not None else time.time(
        )
        self.last_event_type = event_type

    def take_events(self) -> Optional[list[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED_SWAP = enum.auto()
    PREEMPTED_RECOMPUTE = enum.auto()
    PLACEHOLDER_FOR_FINISHED = enum.auto(
    )  # Note: To be used only for determining if a request is finished
    # Note: anything after PLACEHOLDER_FOR_FINISHED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PLACEHOLDER_FOR_FINISHED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
