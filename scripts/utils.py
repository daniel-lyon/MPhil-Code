""" Utility functions for project """

import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap

# make new colourmap with zero points as white
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def scatter_colour_density(fig, x, y, cmap=white_viridis):
    """ Make a scatter plot with density map """
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=cmap)
    fig.colorbar(density, label='Number of points per pixel')