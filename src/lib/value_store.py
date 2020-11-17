class ValueStore:
    """ValueStore

    A value store for learning (input, value)
    - implementation of learning is further specified
    - built-in metrics method
    """

    def __init__(self, name):
        self.name = name

        self.metrics_history = {}

    #
    # metrics history function
    #
    def record(self, metrics_names, log=True):
        for metrics_name in metrics_names:

            metrics_method = getattr(self, metrics_name)
            metrics = metrics_method()

            if metrics_name not in self.metrics_history.keys():
                self.metrics_history[metrics_name] = []

            self.metrics_history[metrics_name].append(metrics)

            if log:
                print(f"{self.name}_{metrics_name}: {metrics:.3f}")

            if metrics_name == "diff":
                self.backup()

    def converged(self, metrics_name, threshold, log=True):
        last_3 = self.metrics_history[metrics_name][-4:-1]

        if len(last_3) < 3:
            return False

        last_3_mean = sum(last_3) / len(last_3)

        if last_3_mean < threshold:
            if log:
                print(f"{self.name}_{metrics_name} has converged")
            return True
        else:
            return False
