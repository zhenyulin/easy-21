from tinygrad.tensor import Tensor
from tinygrad.utils import layer_init_uniform


class MLP:
    def __init__(
        self,
        input_size,
        network_size,
        gpu=True,
        learning_rate=0.001,
    ):
        self.gpu = gpu
        self.input_size = input_size
        self.network_size = network_size

        self.layers = [
            Tensor(layer_init_uniform(in_size, out_size), gpu=self.gpu)
            for (in_size, out_size) in zip(
                [input_size, *network_size[:-1]], network_size
            )
        ]

    #
    # getter functions
    #
    def __call__(self, x):
        output = Tensor(x, gpu=self.gpu)

        for i, layer in enumerate(self.layers):
            output = output.dot(layer)
            if i != len(self.layers) - 1:
                output = output.relu()

        return output

    @property
    def weights(self):
        """
        make sure output data is not in the type of opencl Buffer
        """
        return [layer.cpu().data for layer in self.layers]

    @property
    def flatten_weights(self):
        return [p for layer in self.weights for unit in layer for p in unit]

    def clone(self):
        return [Tensor(layer.cpu().data) for layer in self.layers]

    #
    # setter functions
    #
    def reset_grad(self):
        for layer in self.layers:
            layer.grad = None

    def learn(self, x, y, step_size=0.01):

        _y = Tensor(y, gpu=self.gpu)
        two = Tensor([[2] for _ in range(len(y))], gpu=self.gpu)
        lr = Tensor([[step_size]], gpu=self.gpu)

        output = self.__call__(x)
        loss = (output - _y).pow(two).mean()

        loss.backward()
        for layer in self.layers:
            layer -= layer.grad * lr

        self.reset_grad()
