from vidur.types.base_int_enum import BaseIntEnum


class ReplicaSchedulerType(BaseIntEnum):
    FASTER_TRANSFORMER = 1
    ORCA = 2
    SARATHI = 3
    VLLM = 4
    LIGHTLLM = 5
    VTC = 6
    VTC_SARATHI = 7

