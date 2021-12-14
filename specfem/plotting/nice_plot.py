import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def plotting_image(image, cmap='nice', aspect=0.3, vmin_max=(None, None), newfigure=True, fig_number=1, colorbar=True, figsize=(8,8), save=False, savename='', tform_overlay=(False, 0), title='', extent=None, interpolation='nearest', logscale=False, linthresh=1e-4, cbar_label=''):
    """
    Plotting a 2D image.

    Args:
        image         (array) : a 2d array (image)
        cmap          (str)   : colormap used to plot the image. Defaults to 'nice' = 'RdGy.
        aspect        (scalar): aspect ratio of the axes. Defaults to 0.3.
        vmin_max      (tuple) : (vmin, vmax) min and max values of the colorscale. Defaults to (None, None).
        newfigure     (bool)  : if True a new figure will be created. Defaults to True.
        colorbar      (bool)  : if True a colorbar will be created. Defaults to True.
        figsize       (tuple) : size of the figure. Defaults to (8,8).
        save          (bool)  : if True the figure will be saved as 'savename'. Defaults to False.
        savename      (str)   : file name of the saved figure. Defaults to ''.
        tform_overlay (tuple) : first element if True will plot the second element on top. Defaults to (False, 0).
        title         (str)   : title of the plot. Defaults to ''.
        interpolation (str)   : interpolation method in im.imshow. Defaults to 'nearest'.
        logscale      (bool)  : if True a logscale will be used Defaults to False.
        linthresh     (scalar): plotting parameter. Defaults to 1e-4.
        cbar_label    (str)   : label for the colorbar. Defaults to ''.
    """
    
    if cmap == 'black_red':
        colors_cmap = ['k', 'darkslateblue', 'ivory', 'orangered', 'maroon']
        nodes = [0.0, 0.2, 0.5, 0.8, 1.0]
        cmap = colors.LinearSegmentedColormap.from_list('mycmap', list(zip(nodes,colors_cmap)))

    if cmap == 'nice':
        colors_cmap = ['cyan', 'darkturquoise', 'blue', 'steelblue', 'lightgrey', 'slategray', 'red', 'gold', 'yellow']
        nodes  = [0.0, 0.1, .3, .42, .5, .58, .7, 0.9, 1.0]
        cmap = colors.LinearSegmentedColormap.from_list('mycmap', list(zip(nodes,colors_cmap)))
        
    if newfigure:
        fig, axs = plt.subplots(figsize=figsize, constrained_layout=True)
        fig.number = fig_number
        axs.set_title(title)
        axs.xaxis.set_ticks_position('top')
        axs.xaxis.set_label_position('top')
        
    vmin, vmax = vmin_max
    
    if logscale:
        xrange = np.linspace(extent[0], extent[1], image.shape[1])
        zrange = np.linspace(extent[2], extent[3], image.shape[0])
        X, Z = np.meshgrid(xrange, zrange)
        im = axs.pcolormesh(X, Z, image, 
                            norm = colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=vmin, vmax=vmax, base=10), 
                            cmap = cmap, 
                            shading = 'auto')
    else:
        im = axs.imshow(image, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax, interpolation=interpolation, extent=extent)
    
    if colorbar:
        cbar = fig.colorbar(im, aspect=40, pad=0.0, label=cbar_label, orientation='horizontal')
    
    if save:
        if not os.path.exists('plots/'):
            os.system('mkdir -p plots/')
        path2plots = 'plots/'
        
        if savename:
            savename += '.pdf'
            plt.savefig(os.path.join(path2plots, savename))
        else:
            print('Figure was not saved, savename is empty')