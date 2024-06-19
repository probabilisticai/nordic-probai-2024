import numpy as np
import matplotlib.pyplot as plt
import imageio
from probai.src.evaluation import bond_analyze
from probai.src.data.mini_qm9 import DATASET_INFO


def plot_data3d(
    positions,
    atom_type,
    camera_elev=0,
    camera_azim=0,
    save_path=None,
    spheres_3d=False,
    bg="black",
    alpha=1.0,
):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = "#FFFFFF" if bg == "black" else "#666666"

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect("auto")
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == "black":
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    # ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == "black":
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color)

    max_value = positions.abs().max().item()

    # axis_lim = 3.2
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype("uint8")
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color):
    # draw_sphere(ax, 0, 0, 0, 1)
    # draw_sphere(ax, 1, 1, 1, 1)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    # ax.set_facecolor((1.0, 0.47, 0.42))
    colors_dic = np.array(DATASET_INFO["colors_dic"])
    radius_dic = np.array(DATASET_INFO["radius_dic"])
    area_dic = 1500 * radius_dic**2
    # areas_dic = sizes_dic * sizes_dic * 3.1416

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(
            x, y, z, s=areas, alpha=0.9 * alpha, c=colors
        )  # , linewidths=2, edgecolors='#FFFFFF')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = (
                DATASET_INFO["atom_decoder"][atom_type[i]],
                DATASET_INFO["atom_decoder"][atom_type[j]],
            )
            s = sorted((atom_type[i], atom_type[j]))
            draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
            line_width = (3 - 2) * 2 * 2

            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    linewidth_factor = 1
                ax.plot(
                    [x[i], x[j]],
                    [y[i], y[j]],
                    [z[i], z[j]],
                    linewidth=line_width * linewidth_factor,
                    c=hex_bg_color,
                    alpha=alpha,
                )


def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x + xs,
        y + ys,
        z + zs,
        rstride=2,
        cstride=2,
        color=color,
        linewidth=0,
        alpha=alpha,
    )
