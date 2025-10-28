# Lab 4: Fair scheduling for LLM Inference Workloads

## Environment Setup
1. Clone the repository
2. Create and activate a Python 3.10 virtual environment
    - `python3.10 -m venv .venv`
    - `source .venv/bin/activate`
3. Install the dependencies
    - `pip install -r requirements.txt`

## Overview

This lab focuses on implementing fair scheduling policies in Vidur based on the [Virtual Token Counter (VTC) algorithm](https://www.usenix.org/system/files/osdi24-sheng.pdf). The lab is divided into two major components: implementing basic VTC scheduler and creating a custom fair scheduler combining chunking with VTC.

## Part 1: Basic VTC Scheduler (7.5 pts)

#### Request Class Methods (0.5 points)
- **`get_prefill_token_service_cost()`** (0.25 pts)
  - Calculate standard prefill token service cost as defined in VTC paper
  
- **`get_decode_token_service_cost()`** (0.25 pts)
  - Calculate standard decode token service cost as per VTC paper

#### VTC Core Functions (7 points)
Implement the following methods in `VTCReplicaScheduler`:

- **`_update_counter_on_arrival()`** (1.5 pts)
  - Updates virtual counter for new request arrivals
  - Handle counter lift

- **`_update_counter_on_batch_completion()`** (1 pts)
  - Updates virtual counters post batch processing
  - Handle multi-request batches
  - Handle weighted updates

- **`_select_next_request()`** (1 pts)
  - Implements VTC-based request selection

- **`_get_next_batch()`** (3.5 pts)
  - Manages batch creation and scheduling
  - Augment the prefill prioritizing schedule from vLLM with VTC

## Part 2: VTC Sarathi Scheduler (7.5 points)

#### Core VTC-Sarathi Functions
- **`_update_counter_on_batch_completion()`** (1 pts)
  - Handles chunked request counter updates in addition to existing changes

- **`_select_next_request()`** (2 pts)
  - VTC selection with chunking support

- **`_get_next_batch()`** (4.5 pts)
  - Manages batch creation and scheduling
  - Augment the mixed batching schedule from Sarathi with VTC. Combining VTC with chunking enables preemption and fair-scheduling at a chunk level.

## Resources

- [VTC Paper](https://www.usenix.org/system/files/osdi24-sheng.pdf)
- [Vidur Paper](https://arxiv.org/abs/2405.05465)

## Important Notes

- Follow PEP 8 guidelines
- Include detailed docstrings
- Handle edge cases
