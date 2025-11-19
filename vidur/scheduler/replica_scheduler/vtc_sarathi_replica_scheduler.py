from typing import Dict, Optional, List, Tuple
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
        client_id = request.client_id

        if client_id in self._virtual_counters:
            return

        if self._virtual_counters:
            min_counter = min(self._virtual_counters.values())
            self._virtual_counters[client_id] = min_counter
        else:
            self._virtual_counters[client_id] = 0.0


    def _update_counter_on_batch_completion(self, batch: Batch) -> None:
        """
        Update virtual counters when batch complete processing.
        """
        client_service = defaultdict(float)

        for request, num_tokens in zip(batch.requests, batch.num_tokens):
            client_id = request.client_id

            tokens_after = request.num_processed_tokens
            tokens_before = tokens_after - num_tokens

            prefill_tokens_remaining = max(request.num_prefill_tokens - tokens_before, 0)
            prefill_tokens = min(prefill_tokens_remaining, num_tokens)
            decode_tokens = num_tokens - prefill_tokens

            cost = (
                request.get_prefill_token_service_cost(prefill_tokens)
                + request.get_decode_token_service_cost() * decode_tokens
            )

            damping = (
                self._config.client_zero_damping
                if client_id == 0
                else self._config.client_one_damping
            )

            client_service[client_id] += cost / damping

        for client_id, service in client_service.items():
            old_counter = self._virtual_counters.get(client_id, 0.0)
            self._virtual_counters[client_id] = old_counter + service


    def _select_next_request(self, running_prefills: List[Request]) -> Tuple[Request, int, bool]:
        """
        Select next request based on VTC algorithm.
        Return:
        - Request: Request to schedule
        - Request index: Index of request in the waiting queue/running prefills list
        - Is running prefill: Whether the request is a running prefill
        """
        candidates = []

        for idx, req in enumerate(running_prefills):
            counter = self._virtual_counters.get(req.client_id, 0.0)
            candidates.append((counter, req.arrived_at, idx, req, True))

        for idx, req in enumerate(self._request_queue):
            counter = self._virtual_counters.get(req.client_id, 0.0)
            candidates.append((counter, req.arrived_at, idx, req, False))

        if not candidates:
            return None, -1, False

        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, idx, request, is_running = candidates[0]

        self._last_client_scheduled = request.client_id

        return request, idx, is_running

    def _get_next_batch(self) -> Batch:
        """
        Get next batch to schedule.
        """
        requests = []
        num_tokens = []
        skipped_requests = []
        running_prefills = []
        contains_prefill = False
        num_batch_tokens = 0

        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    break
            else:
                self._allocate_request(request)
                assert request.is_prefill_complete
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        while running_prefills or self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                break
            if len(requests) == self._max_micro_batch_size:
                break

            request, idx, is_running = self._select_next_request(running_prefills)

            if request is None:
                break

            if is_running:
                running_prefills.pop(idx)
            else:
                self._request_queue.pop(idx)

            if not is_running and not self._can_allocate_request(request):
                self._request_queue.insert(0, request)
                break

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                continue

            if not is_running:
                self._allocate_request(request)

            contains_prefill = contains_prefill or not request.is_prefill_complete
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        self._preempted_requests = skipped_requests + running_prefills + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests,
            key=lambda req: (self._virtual_counters.get(req.client_id, 0.0), req.arrived_at),
        )

        if not requests:
            return None

        return Batch(self._replica_id, requests, num_tokens)
