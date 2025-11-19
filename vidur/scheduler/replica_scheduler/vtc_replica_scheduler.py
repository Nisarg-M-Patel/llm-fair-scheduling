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

        print(f"\n=== Batch Completion Debug ===")
        for request, num_tokens in zip(batch.requests, batch.num_tokens):
            print(f"Request {request.id}: client={request.client_id}, "
                f"processed={request.num_processed_tokens}, batch_tokens={num_tokens}, "
                f"is_prefill_complete={request.is_prefill_complete}, "
                f"prefill_tokens={request.num_prefill_tokens}")
        print(f"Counters before: {dict(self._virtual_counters)}")
        
        client_service = {}

        for request, num_tokens in zip(batch.requests, batch.num_tokens):
            client_id = request.client_id
            
            if client_id not in client_service:
                client_service[client_id] = 0.0
            
            # Determine what phase was processed DURING this batch
            # tokens_before_batch = current position - tokens just processed
            tokens_before_batch = request.num_processed_tokens - num_tokens
            
            # Was the request in prefill phase when batch started?
            if tokens_before_batch < request.num_prefill_tokens:
                # Was processing prefill tokens
                service_cost = request.get_prefill_token_service_cost(num_tokens)
            else:
                # Was processing decode tokens
                service_cost = num_tokens * request.get_decode_token_service_cost()
            
            client_service[client_id] += service_cost
        
        for client_id, service in client_service.items():
            self._virtual_counters[client_id] += service
        
        print(f"Counters after: {dict(self._virtual_counters)}")
        
        


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
        
        if self._request_queue:
            print(f"\n=== Batch Formation (Prefill Queue) ===")
            print(f"Queue size: {len(self._request_queue)}")
            for idx, req in enumerate(self._request_queue[:20]):
                counter = self._virtual_counters.get(req.client_id, 0.0)
                print(f"  [{idx}] req_id={req.id}, client={req.client_id}, counter={counter:.2f}")
        
        # Sort queue by virtual counter to try in order
        sorted_indices = sorted(
            range(len(self._request_queue)), 
            key=lambda i: self._virtual_counters.get(self._request_queue[i].client_id, 0.0)
        )
        
        attempted = set()
        
        for queue_idx in sorted_indices:
            # Check if this request is still in queue (we might have already taken it)
            if queue_idx >= len(self._request_queue):
                continue
                
            request = self._request_queue[queue_idx]
            
            # Skip if already attempted
            if request.id in attempted:
                continue
            attempted.add(request.id)
            
            # Check batch size limits
            if len(requests) == self._max_micro_batch_size:
                break
            
            if len(self._allocation_map) == self._config.batch_size_cap:
                break
            
            next_num_tokens = self._get_request_next_num_tokens(request)
            
            # Check if request can be allocated
            if not self._can_allocate_request(request):
                print(f"  → SKIP: Can't allocate req {request.id}")
                continue
            
            # Check if adding this request would exceed batch token limit
            new_num_tokens = num_tokens + [next_num_tokens]
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens)
            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                print(f"  → SKIP: Batch tokens would exceed ({new_num_batch_tokens} > {self._config.max_tokens_in_batch})")
                continue
            
            # This request fits! Remove from queue and add to batch
            # Find current index since it might have shifted
            current_idx = next(i for i, r in enumerate(self._request_queue) if r.id == request.id)
            request = self._request_queue.pop(current_idx)
            
            print(f"  → ADDED req {request.id} to batch")
            
            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens
        
        if requests:
            print(f"Returning prefill batch with {len(requests)} requests: {[r.id for r in requests]}")
            return Batch(self._replica_id, requests, num_tokens)
        
        # Handle preempted requests (decode phase)
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
        
        print(f"Returning decode batch with {len(requests)} requests: {[r.id for r in requests]}")
        return Batch(self._replica_id, requests, num_tokens)