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
            min_counter = min(self._virtual_counters.values())
            self._virtual_counters[client_id] = min_counter
        else:
            self._virtual_counters[client_id] = 0.0

    def _update_counter_on_batch_completion(self, batch: Batch) -> None:
        """
        Update virtual counters when batch complete processing.
        """
        client_service = {}
    
        for request, num_tokens in zip(batch.requests, batch.num_tokens):
            client_id = request.client_id
            
            if client_id not in client_service:
                client_service[client_id] = 0.0
            
            if not request.is_prefill_complete:
                service_cost = request.get_prefill_token_service_cost(num_tokens)
            else:
                service_cost = num_tokens * request.get_decode_token_service_cost()
            
            client_service[client_id] += service_cost
        
        for client_id, service in client_service.items():
            self._virtual_counters[client_id] += service



    def _select_next_request(self) -> Tuple[Request, int]:
        """
        Select next request based on VTC algorithm.
        Return:
        - Request: Request to schedule
        - Request index: Index of request in the waiting queue
        """
        if not self._request_queue:
            return None, None
    
        min_counter = float('inf')
        selected_request = None
        selected_index = None
        
        for idx, request in enumerate(self._request_queue):
            client_id = request.client_id
            counter = self._virtual_counters.get(client_id, 0.0)
            
            if counter < min_counter:
                min_counter = counter
                selected_request = request
                selected_index = idx
        
        return selected_request, selected_index

    def _get_next_batch(self) -> Batch:
        """
        Get next batch to schedule.
        """
        requests = []
        num_tokens = []
        num_batch_tokens = 0
        
        while self._request_queue:
            request, idx = self._select_next_request()
            
            if request is None:
                break
            
            next_num_tokens = self._get_request_next_num_tokens(request)
            
            if not self._can_allocate_request(request):
                break
            
            new_num_tokens = num_tokens + [next_num_tokens]
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                break
            
            if len(self._allocation_map) == self._config.batch_size_cap:
                break
            
            if len(requests) == self._max_micro_batch_size:
                break
            
            request = self._request_queue.pop(idx)
            
            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens
        
        if requests:
            return Batch(self._replica_id, requests, num_tokens)
        
        self._preempted_requests.sort(key=lambda r: r.arrived_at)
        
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break
            
            min_counter = float('inf')
            selected_idx = None
            
            for idx, request in enumerate(self._preempted_requests):
                counter = self._virtual_counters.get(request.client_id, 0.0)
                if counter < min_counter:
                    min_counter = counter
                    selected_idx = idx
            
            if selected_idx is None:
                break
                
            request = self._preempted_requests.pop(selected_idx)
            
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
