from vidur.entities import Request
from vidur.config import FairServiceCalculatorConfig

class FairServiceCalculator:
    """
    Service rate calculator to calculate the service rate of a given request
    """

    def __init__(self, config: FairServiceCalculatorConfig):
        self._config = config

    def calculate_service_for_request(self, request: Request) -> float:
        """
        Calculate the service rate of the request
        :return: service rate
        """
        return (
            self.request.num_prefill_tokens * self._config.prefill_cost_factor
            + self.request.num_decode_tokens * self._config.decode_cost_factor
        )
