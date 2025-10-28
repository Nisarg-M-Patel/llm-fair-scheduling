from abc import ABC, abstractmethod
from vidur.config import BaseClientAssignmentGeneratorConfig


class BaseClientAssignmentGenerator(ABC):
    def __init__(self, config: BaseClientAssignmentGeneratorConfig):
        self._config = config

    @abstractmethod
    def get_next_client_id(self) -> str:
        pass
