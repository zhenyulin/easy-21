from .metrics import Metrics


class ValueStore:
    """ValueStore

    A value store for learning (input, value)
    - implementation of learning is further specified
    - built-in metrics method
    """

    def __init__(self, name):
        self.name = name
        self.metrics = Metrics(name)
