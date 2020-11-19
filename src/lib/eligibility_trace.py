class EligibilityTrace:
    def __init__(self):
        self.data = {}

    #
    # utility functions
    #
    def init_if_not_found(self, key):
        if key not in self.data.keys():
            self.data[key] = 0

    #
    # getter functions
    #
    def get(self, key):
        self.init_if_not_found(key)
        return self.data[key]

    def keys(self):
        return self.data.keys()

    #
    # setter functions
    #
    def decay(self, discount=1, lambda_value=1):
        # decay the frequency weights if not visited
        for key in self.data.keys():
            self.data[key] *= discount * lambda_value

    def update(self, key, discount=1, lambda_value=1):
        self.decay(discount, lambda_value)
        self.init_if_not_found(key)
        self.data[key] += 1

    def reset(self):
        self.data = {}
