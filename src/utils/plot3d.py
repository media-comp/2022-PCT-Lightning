import matplotlib.pyplot as plt


def plot3d(xyz, savepath, label=""):
    point_x = xyz[:, 0]
    point_y = xyz[:, 1]
    point_z = xyz[:, 2]
    point_size = [10] * len(point_x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_x, point_y, point_z, s=point_size, c="r", marker="o")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.text2D(0.05, 0.05, label, transform=ax.transAxes, color="blue")
    plt.savefig(savepath, bbox_inches="tight")
