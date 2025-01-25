"""Defines Joe's custom colormap and registers it with matplotlib.

This module must but imported before the 'jy_pink' colormap can be used.
"""
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def normalize_color(color):
    return [i / 255 for i in color]


def make_color_dict(colors):
    color_dict = {
        "red": [],
        "blue": [],
        "green": [],
    }
    N = len(colors) - 1
    for i, color in enumerate(colors):
        if isinstance(color, str):
            color = hex_color_to_rgb(color)
        if any(i > 1 for i in color):
            color = normalize_color(color)
        for j, c in enumerate(["red", "green", "blue"]):
            color_dict[c].append((i / N, color[j], color[j]))
    return color_dict


def register_color_map(cmap_name, colors):
    if cmap_name in plt.colormaps():
        return
    cdict = make_color_dict(colors)
    cmap = LinearSegmentedColormap(cmap_name, cdict)
    colormaps.register(cmap=cmap)


color_pts = [
    (45, 96, 114),
    (243, 210, 181),
    (242, 184, 164),
    (242, 140, 143),
    (208, 127, 127),
]


def register():
    register_color_map("jy_pink", color_pts)
    register_color_map("jy_pink_r", list(reversed(color_pts)))
