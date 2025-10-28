from vidur.request_generator.base_client_assignment_generator import BaseClientAssignmentGenerator


class FixedClientAssignmentGenerator(BaseClientAssignmentGenerator):
    def get_next_client_id(self):
        return 0
