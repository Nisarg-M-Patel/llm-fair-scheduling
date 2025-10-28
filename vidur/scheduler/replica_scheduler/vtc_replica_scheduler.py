from typing import Tuple, Dict
from collections import defaultdict

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import VLLMReplicaScheduler


class VTCReplicaScheduler(VLLMReplicaScheduler):
    def __init__(self, *args, **kwargs):
        """
        VTC scheduler (using prefill-prioritizing schedule) that provides fairness guarantees.
        """
        super().__init__(*args, **kwargs)

        self._virtual_counters: Dict[int, float] = defaultdict(float)
        self._last_client_scheduled: int = None

    def on_batch_end(self, batch: Batch) -> None:
        self._update_counter_on_batch_completion(batch)
        super().on_batch_end(batch)

    def add_request(self, request: Request) -> None:
        """Add new request and update VTC state."""
        self._update_counter_on_arrival(request)
        super().add_request(request)

    def _update_counter_on_arrival(self, request: Request) -> None:
        """
        Update virtual counter when a new request arrives.
        """
        raise NotImplementedError

    def _update_counter_on_batch_completion(self, batch: Batch) -> None:
        """
        Update virtual counters when batch complete processing.
        """
        raise NotImplementedError


    def _select_next_request(self) -> Tuple[Request, int]:
        """
        Select next request based on VTC algorithm.
        Return:
        - Request: Request to schedule
        - Request index: Index of request in the waiting queue
        """
        raise NotImplementedError

    def _get_next_batch(self) -> Batch:
        """
        Get next batch to schedule.
        """
        raise NotImplementedError
