import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as ln
import matplotlib.path
import matplotlib.patches

plt.style.use('seaborn-whitegrid')


def geom(coord, jn, name):
    """Геометрия."""

    x = coord[:, 0]
    y = coord[:, 1]
    fig, ax = plt.subplots(num=name)
    ax.set_aspect('equal')
    ax.scatter(x, y, s=20, c='blue', alpha=0.5)
    for i in range(len(jn)):
        line = ln.Line2D([jn[i, 0, 0], jn[i, 1, 0]], [jn[i, 0, 1], jn[i, 1, 1]], color='green')
        ax.add_line(line)
        ax.annotate(str(i), ((jn[i, 0, 0] + jn[i, 1, 0]) / 2, (jn[i, 0, 1] + jn[i, 1, 1]) / 2), size=12, xytext=(
            0, 0), ha='right', c='red', textcoords='offset points')
    for i in range(len(coord)):
        ax.annotate(str(i), (x[i], y[i]), size=12, xytext=(
            0, 0), ha='right', c='blue', textcoords='offset points')

    plt.title(name, pad=20)
    ax.set_xlabel('X, м')
    ax.set_ylabel('Z, м')
    plt.show()


if __name__ == '__main__':
    print(geom.__doc__)
    input('Press Enter:')
