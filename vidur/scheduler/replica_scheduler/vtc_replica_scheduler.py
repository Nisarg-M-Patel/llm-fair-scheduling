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
        client_id = request.client_id

        if client_id in self._virtual_counters:
            return

        if self._virtual_counters:
            self._virtual_counters[client_id] = min(self._virtual_counters.values())
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

            service_cost = (
                request.get_prefill_token_service_cost(prefill_tokens)
                + request.get_decode_token_service_cost() * decode_tokens
            )

            damping = (
                self._config.client_zero_damping
                if client_id == 0
                else self._config.client_one_damping
            )

            client_service[client_id] += service_cost / damping

        for client_id, service in client_service.items():
            self._virtual_counters[client_id] += service

        if batch.requests:
            self._last_client_scheduled = batch.requests[-1].client_id

    def _select_next_request(self) -> Tuple[Request, int]:
        """
        Select next request based on VTC algorithm.
        Return:
        - Request: Request to schedule
        - Request index: Index of request in the waiting queue
        """
        if not self._request_queue:
            return None, None

        best_idx = None
        best_key = None

        for idx, request in enumerate(self._request_queue):
            counter = self._virtual_counters.get(request.client_id, 0.0)
            key = (counter, request.arrived_at)

            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx

        selected_request = self._request_queue[best_idx]
        self._last_client_scheduled = selected_request.client_id

        return selected_request, best_idx

    def _get_next_batch(self) -> Batch:
        """
        Get next batch to schedule.
        """
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        while self._request_queue:
            if len(requests) == self._max_micro_batch_size:
                break
            if len(self._allocation_map) == self._config.batch_size_cap:
                break

            request, idx = self._select_next_request()
            if request is None:
                break

            next_num_tokens = self._get_request_next_num_tokens(request)

            prospective_tokens = num_tokens + [next_num_tokens]
            if prospective_tokens:
                prospective_batch_tokens = len(prospective_tokens) * max(prospective_tokens)
                if prospective_batch_tokens > self._config.max_tokens_in_batch:
                    break

            if not self._can_allocate_request(request):
                break

            self._request_queue.pop(idx)
            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens

        if requests:
            return Batch(self._replica_id, requests, num_tokens)

        self._preempted_requests.sort(
            key=lambda r: (self._virtual_counters.get(r.client_id, 0.0), r.arrived_at)
        )

        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.pop(0)

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
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

        if not requests:
            return None

        return Batch(self._replica_id, requests, num_tokens)
