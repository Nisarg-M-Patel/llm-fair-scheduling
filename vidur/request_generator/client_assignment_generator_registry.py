from vidur.request_generator.fixed_client_assignment_generator import (
    FixedClientAssignmentGenerator,
)
from vidur.request_generator.binomial_client_assignment_generator import (
    BinomialClientAssignmentGenerator,
)
from vidur.types import ClientAssignmentGeneratorType
from vidur.utils.base_registry import BaseRegistry


class ClientAssignmentGeneratorRegistry(BaseRegistry):
    pass


ClientAssignmentGeneratorRegistry.register(
    ClientAssignmentGeneratorType.FIXED, FixedClientAssignmentGenerator
)
ClientAssignmentGeneratorRegistry.register(
    ClientAssignmentGeneratorType.BINOMIAL, BinomialClientAssignmentGenerator
)

