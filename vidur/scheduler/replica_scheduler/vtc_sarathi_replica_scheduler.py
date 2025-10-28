from typing import Dict, Optional, List
from collections import defaultdict

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import SarathiReplicaScheduler


class VTCSarathiReplicaScheduler(SarathiReplicaScheduler):
    """
    VTC-based Sarathi scheduler that provides fairness guarantees at chunk level.
    Inherits chunked prefill logic from SarathiReplicaScheduler.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # VTC-specific attributes
        self._virtual_counters: Dict[int, float] = defaultdict(float)
        self._last_client_scheduled: Optional[int] = None

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
        This can be identical to the standard VTC implementation.
        """
        raise NotImplementedError

    def _update_counter_on_batch_completion(self, batch: Batch) -> None:
        """
        Update virtual counters when batch complete processing.
        """
        raise NotImplementedError

    def _select_next_request(self, running_prefills: List[Request]) -> List[Request, int, bool]:
        """
        Select next request based on VTC algorithm.
        Return:
        - Request: Request to schedule
        - Request index: Index of request in the waiting queue/running prefills list
        - Is running prefill: Whether the request is a running prefill
        """
        raise NotImplementedError

    def _get_next_batch(self) -> Batch:
        """
        Get next batch to schedule.
        """
        raise NotImplementedError
