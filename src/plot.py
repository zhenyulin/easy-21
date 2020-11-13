import numpy as np
import matplotlib.pyplot as plt

# TODO: create an animation to show how the surface changes
def plot_2d_value_map(value_map, name, episodes_count):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    dealer = np.arange(1, 11, 1)
    player = np.arange(1, 22, 1)
    X, Y = np.meshgrid(dealer, player)

    Z = np.array([[value_map.get((x, y)) for x in dealer] for y in player])

    plt.xticks(dealer)
    plt.yticks(player)
    ax.set_xlabel("Dealer")
    ax.set_ylabel("Player")
    ax.set_zlabel("Value")
    plt.title(f"{name}, episode count: {episodes_count:.0e}")

    ax.plot_surface(X, Y, Z)


def plot_line(data):
    plt.figure(figsize=(12, 12))
    ax = plt.axes()
    x = np.arange(1, len(data) + 1, 1)
    ax.plot(x, data)
