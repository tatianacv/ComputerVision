from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1 # For colorbars

from scipy.ndimage import grey_dilation

from skimage import img_as_float
from skimage import color
from skimage import exposure
from skimage.util.dtype import dtype_limits

from matplotlib.colors import NoNorm, BoundaryNorm, ListedColormap
from matplotlib import cm
from matplotlib.cm import ScalarMappable, get_cmap


__all__ = ['imshow_all', 'imshow_with_histogram', 'mean_filter_demo',
           'mean_filter_interactive_demo', 'plot_cdf', 'plot_histogram', 'colorbars','add_colorbar','match_axes_height', 'scatter_matrix',
           'discrete_cmap', 
           'discrete_colorbar', 'show_segmentation', 'show_features']


# Gray-scale images should actually be gray!
plt.rcParams['image.cmap'] = 'gray'


#--------------------------------------------------------------------------
#  Custom `imshow` functions
#--------------------------------------------------------------------------

def imshow_rgb_shifted(rgb_image, shift=100, ax=None):
    """Plot each RGB layer with an x, y shift."""
    if ax is None:
        ax = plt.gca()

    height, width, n_channels = rgb_image.shape
    x = y = 0
    for i_channel, channel in enumerate(iter_channels(rgb_image)):
        image = np.zeros((height, width, n_channels), dtype=channel.dtype)

        image[:, :, i_channel] = channel
        ax.imshow(image, extent=[x, x+width, y, y+height], alpha=0.7)
        x += shift
        y += shift
    # `imshow` fits the extents of the last image shown, so we need to rescale.
    ax.autoscale()
    ax.set_axis_off()


def imshow_all(*images, **kwargs):
    """ Plot a series of images side-by-side.

    Convert all images to float so that images have a common intensity range.

    Parameters
    ----------
    limits : str
        Control the intensity limits. By default, 'image' is used set the
        min/max intensities to the min/max of all images. Setting `limits` to
        'dtype' can also be used if you want to preserve the image exposure.
    titles : list of str
        Titles for subplots. If the length of titles is less than the number
        of images, empty strings are appended.
    kwargs : dict
        Additional keyword-arguments passed to `imshow`.
    """
    images = [img_as_float(img) for img in images]

    titles = kwargs.pop('titles', [])
    if len(titles) != len(images):
        titles = list(titles) + [''] * (len(images) - len(titles))

    limits = kwargs.pop('limits', 'image')
    if limits == 'image':
        kwargs.setdefault('vmin', min(img.min() for img in images))
        kwargs.setdefault('vmax', max(img.max() for img in images))
    elif limits == 'dtype':
        vmin, vmax = dtype_limits(images[0])
        kwargs.setdefault('vmin', vmin)
        kwargs.setdefault('vmax', vmax)

    nrows, ncols = kwargs.get('shape', (1, len(images)))

    size = nrows * kwargs.pop('size', 5)
    width = size * len(images)
    if nrows > 1:
        width /= nrows * 1.33
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, size))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, **kwargs)
        ax.set_title(label)


def imshow_with_histogram(image, xlim=None, **kwargs):
    """ Plot an image side-by-side with its histogram.

    - Plot the image next to the histogram
    - Plot each RGB channel separately (if input is color)
    - Automatically flatten channels
    - Select reasonable bins based on the image's dtype

    See `plot_histogram` for information on how the histogram is plotted.
    """
    width, height = plt.rcParams['figure.figsize']
    fig, (ax_image, ax_hist) = plt.subplots(ncols=2, figsize=(2*width, height))

    kwargs.setdefault('cmap', plt.cm.gray)
    ax_image.imshow(image, **kwargs)
    plot_histogram(image, ax=ax_hist, xlim=xlim)

    # pretty it up
    ax_image.set_axis_off()
    match_axes_height(ax_image, ax_hist)
    return ax_image, ax_hist


#--------------------------------------------------------------------------
#  Helper functions
#--------------------------------------------------------------------------

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.
    
    See https://stackoverflow.com/a/33505522"""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def colorbars(axes=None, fig=None, return_handles=False, 
              aspect=20, pad_fraction=0.5, **kwargs):
    # Add colorbars to all axes that contain an image
    # axes: Axes object or list of Axes
    # fig: if axes is None, figure from which to extract axes
    # Returns a list of Colorbar objects
    if (axes==None):
        if (fig==None):
            fig = plt.gcf()
        axes=fig.axes
    if (type(axes) is list):
        cbars=[]
        for ax in plt.gcf().axes:
            cbars.extend(colorbars(ax, return_handles=True))
    else:
        imgs=axes.images
        if (len(imgs)>0):
            #return [plt.colorbar(imgs[0],ax=axes)] # Use first image from the axes
            cbars=[add_colorbar(imgs[0],aspect=aspect, pad_fraction=pad_fraction, **kwargs)]
        else:
            cbars=[]
    if return_handles: 
        return cbars
    else: 
        return

def match_axes_height(ax_src, ax_dst):
    """ Match the axes height of two axes objects.

    The height of `ax_dst` is synced to that of `ax_src`.
    """
    # HACK: plot geometry isn't set until the plot is drawn
    plt.draw()
    dst = ax_dst.get_position()
    src = ax_src.get_position()
    ax_dst.set_position([dst.xmin, src.ymin, dst.width, src.height])


def plot_cdf(image, ax=None, xlim=None):
    img_cdf, bins = exposure.cumulative_distribution(image)
    ax.plot(bins, img_cdf, 'r')
    ax.set_ylabel("Fraction of pixels below intensity")
    if (xlim is None):
        if (image.dtype=='uint8'):
            ax.set_xlim(0,255)
    else:
        ax.set_xlim(xlim[0],xlim[1])


def plot_histogram(image, ax=None, xlim=None, **kwargs):
    """ Plot the histogram of an image (gray-scale or RGB) on `ax`.

    Calculate histogram using `skimage.exposure.histogram` and plot as filled
    line. If an image has a 3rd dimension, assume it's RGB and plot each
    channel separately.
    """
    ax = ax if ax is not None else plt.gca()
    

    if image.ndim == 2:
        _plot_histogram(ax, image, color='black', **kwargs)
    elif image.ndim == 3:
        # `channel` is the red, green, or blue channel of the image.
        for channel, channel_color in zip(iter_channels(image), 'rgb'):
            _plot_histogram(ax, channel, color=channel_color, **kwargs)

    if (xlim is None):
        if (image.dtype=='uint8'):
            ax.set_xlim(0,255)
    else:
        ax.set_xlim(xlim[0],xlim[1])


def _plot_histogram(ax, image, alpha=0.3, **kwargs):
    # Use skimage's histogram function which has nice defaults for
    # integer and float images.
    hist, bin_centers = exposure.histogram(image)
    ax.fill_between(bin_centers, hist, alpha=alpha, **kwargs)
    ax.set_xlabel('intensity')
    ax.set_ylabel('# pixels')


def iter_channels(color_image):
    """Yield color channels of an image."""
    # Roll array-axis so that we iterate over the color channels of an image.
    for channel in np.rollaxis(color_image, -1):
        yield channel


#--------------------------------------------------------------------------
#  Convolution Demo
#--------------------------------------------------------------------------

def mean_filter_demo(image, vmax=1):
    mean_factor = 1.0 / 9.0  # This assumes a 3x3 kernel.
    iter_kernel_and_subimage = iter_kernel(image)

    image_cache = []

    def mean_filter_step(i_step):
        while i_step >= len(image_cache):
            filtered = image if i_step == 0 else image_cache[-1][1]
            filtered = filtered.copy()

            (i, j), mask, subimage = iter_kernel_and_subimage.next()
            filter_overlay = color.label2rgb(mask, image, bg_label=0,
                                             colors=('yellow', 'red'))
            filtered[i, j] = np.sum(mean_factor * subimage)
            image_cache.append((filter_overlay, filtered))

        imshow_all(*image_cache[i_step], vmax=vmax)
        plt.show()
    return mean_filter_step


def mean_filter_interactive_demo(image):
    from IPython.html import widgets
    mean_filter_step = mean_filter_demo(image)
    step_slider = widgets.IntSliderWidget(min=0, max=image.size-1, value=0)
    widgets.interact(mean_filter_step, i_step=step_slider)


def iter_kernel(image, size=1):
    """ Yield position, kernel mask, and image for each pixel in the image.

    The kernel mask has a 2 at the center pixel and 1 around it. The actual
    width of the kernel is 2*size + 1.
    """
    width = 2*size + 1
    for (i, j), pixel in iter_pixels(image):
        mask = np.zeros(image.shape, dtype='int16')
        mask[i, j] = 1
        mask = grey_dilation(mask, size=width)
        mask[i, j] = 2
        subimage = image[bounded_slice((i, j), image.shape[:2], size=size)]
        yield (i, j), mask, subimage


def iter_pixels(image):
    """ Yield pixel position (row, column) and pixel intensity. """
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            yield (i, j), image[i, j]


def bounded_slice(center, xy_max, size=1, i_min=0):
    slices = []
    for i, i_max in zip(center, xy_max):
        slices.append(slice(max(i - size, i_min), min(i + size + 1, i_max)))
    return slices
    
#-------------------------------------------------------------------------
# Classification demo
#-------------------------------------------------------------------------

from matplotlib.colors import Normalize, colorConverter

def discrete_cmap(N=8, colors=None, cmap='viridis', norm=None, 
                  use_bounds=False, zero=None):
    if (colors is None):
        if (norm is None):
            if (use_bounds):
                norm=Normalize(vmin=0, vmax=N-1)
            else:
                norm=Normalize(vmin=-0.5, vmax=N-0.5)
        sm=cm.ScalarMappable(cmap=cmap,norm=norm)
        cols=sm.to_rgba(range(N))[:,:3]
    else:
        cols=colors
        N=len(colors) # Number of rows
    
    if (zero is not None):
        zcol=colorConverter.to_rgb(zero)
        cols=np.insert(cols,0,zcol, axis=0)
    return ListedColormap(cols)

def discrete_colorbar(colors, labels=None, ax=None, cax=None):
    """
        Add a discrete colorbar with custom colors to current axes.
        
        Parameters:
        
        colors: list of RGB tuple
        
        labels: list of tick labels
            assume ticks are at range(len(labels))
    """
    N=len(colors)
    vmin=0; vmax=N-1;
    norm = BoundaryNorm(np.arange(-0.5+vmin,vmax+1.5,1), N)
    s=ScalarMappable(norm=norm, cmap=ListedColormap(colors))
    s.set_array(range(N))
    cb=plt.colorbar(mappable=s, ticks=range(vmin,vmax+1), ax=ax, cax=cax)
    cb.set_norm(norm)
    plt.clim(-0.5+vmin, vmax + 0.5)
    cbticks = plt.getp(cb.ax.axes, 'yticklines')
    plt.setp(cbticks, visible=False)
    if (labels is not None):
        cb.set_ticklabels(labels)
    return cb
    
def scatter_matrix(data, c=None, labels=None, title=None, cmap='viridis', norm=None,
                  figsize=(6,4), show_colorbar=False, class_labels=None):
    '''
    Equivalent of pandas.scatter_matrix or seaborne.pairplot
    '''
    from matplotlib import cm
    nb_features = data.shape[1]
    if (labels is None):
        labels=['Feature {}'.format(i) for i in range(nb_features)]
    if (len(labels)!=nb_features):
        raise ValueError('labels of length {} should have same size as data rows {}'.format(len(labels),nb_features))
    if (data.shape[0]<data.shape[1]):
        print('Warning: multi_scatter received data of shape: nb_samples={}, nb_features={}. If not as intended, please transpose data'.format(data.shape[0],data.shape[1]))
        
    fig, axes = plt.subplots(nb_features,nb_features, figsize=figsize, sharex='col', #sharey='row', 
                             gridspec_kw=dict(hspace=0.05,wspace=0.05))
    
    if (c is None):
        c = np.zeros(data.shape[0],dtype=int)
    nb_classes = np.max(c)+1
    sm=cm.ScalarMappable(cmap=cmap,norm=norm)
    cols=sm.to_rgba(range(nb_classes))[:,:3]
    listedNorm=Normalize(vmin=-0.5, vmax=nb_classes-0.5)
    for i in range(nb_features):
        for j in range(nb_features):
            plt.sca(axes[i,j])
            if (i==j):
                vmin=np.min(data[:,i])
                vmax=np.max(data[:,i])
                for k in range(nb_classes):
                    h, edges = np.histogram(data[c==k,i],range=(vmin,vmax))
                    #plt.fill_between((edges[:-1]+edges[1:])/2, h, color=cols[k,:], alpha=0.5)
                    xs = np.zeros(2*(edges.size-1))
                    ys = np.zeros(2*(edges.size-1))
                    xs[::2]=edges[:-1]; xs[1::2]=edges[1:]
                    ys[::2]=h; ys[1::2]=h;
                    xs=np.insert(xs,0,xs[0]); xs=np.append(xs,xs[-1])
                    ys=np.insert(ys,0,0); ys=np.append(ys,0)
                    plt.fill_between(xs, ys, color=cols[k,:], alpha=0.2)
                    plt.plot(xs, ys, color=cols[k,:])
                    #plt.bar(edges[:-1], h, width=edges[1:]-edges[:-1], color=cols[k,:], alpha=0.5)
                    r=vmax-vmin
                    plt.xlim(vmin-r/20,vmax+r/20)
            else:
                plt.scatter(data[:,j], data[:,i], c=c, axes=axes[i,j], 
                            cmap=ListedColormap(cols), norm=listedNorm, marker='+')
            #if (i<nb_features-1): plt.xticks([])
            if (j>0 and j<nb_features-1): axes[i,j].tick_params(labelleft='off')   
            if (i==j): plt.yticks([]) 
            if (i==nb_features-1 and j==nb_features-1): axes[i,j].tick_params(labelleft='off')   
            if (j==nb_features-1 and i<nb_features-1): axes[i,j].yaxis.tick_right()
            if (i==0): axes[i,j].xaxis.tick_top()
            if (i==nb_features-1): 
                plt.xlabel(labels[j], fontsize=7);
            if (j==0): 
                plt.ylabel(labels[i], fontsize=7);
            plt.tick_params(labelsize=6)
            
    if title is not None:
        plt.suptitle(title,y=0.99)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.90, wspace=0, hspace=0)
    else:
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
    if (show_colorbar):
        plt.subplots_adjust(right=0.95)
        cb=discrete_colorbar(cols, labels=class_labels, ax=axes.ravel().tolist())
        
    return fig


def show_segmentation(im, segm, colors=None, labels=None, figsize=(12,2.5), show_contours=False, cmap=None):
    """
    Show image along with segmentation
    
    Parameters:
    ----------
    
    im : ndarray
        image to use as background
        
    segm : ndarray
        2D label image
    
    colors: list of colors 
        each color is given as a list of R,G,B float values
    """
    fig, axes = plt.subplots(1,3,figsize=figsize)
    
    if colors is None:
        cbcmap=get_cmap('jet')
        N=np.max(segm.ravel())
        colors_i = np.concatenate(([0.], np.linspace(1./N*0.5, 1-(1./N*0.5), N)))
        colors = cbcmap(colors_i)
        colors[0,:]=[0,0,0, 1.]

    plt.sca(axes[0])
    if (show_contours):
        from skimage import segmentation
        #plt.imshow(segmentation.mark_boundaries(im, segm),interpolation='bilinear')
        plt.imshow(im)
        for i,c in enumerate(colors[1:]):
            plt.contour(filters.gaussian(segm==(i+1),1), colors=[c], levels=[0.95])
    else:
        plt.imshow(im, cmap=cmap)
    plt.title('Input image')
    
    plt.sca(axes[1])
    plt.imshow(color.label2rgb(segm, im, colors=colors, alpha=0.5))
    plt.title('Overlay')

    plt.sca(axes[2])
    plt.imshow(color.label2rgb(segm, None, colors=colors), cmap=ListedColormap(colors))
    plt.title('Segmentation')
    
    cb=discrete_colorbar(colors, labels, ax=axes.ravel().tolist())

    import skdemo
    skdemo.match_axes_height(axes[2], cb.ax)

def show_features(features, labels=None, nx='auto', axsize=(4/3,1.1), cmap='gray'):
    """
    Parameters
    ----------
    
    features : ndarray
        stack of feature maps of shape (nb_rows, nb_cols, nb_features)
        
    nx : 'auto' or int
        Number of axes per subplot row
    
    axsize: (sx,sy) 
        size of each image axes
    """
    features = np.atleast_3d(features)
    nb_features = features.shape[2]
    if (nx=='auto'):
        nx=min(nb_features,8)
    ny=(nb_features+nx-1)//nx # Ceil nb_features/nx
    fig, axes = plt.subplots(ny,nx, squeeze=False, figsize=(axsize[0]*nx,axsize[1]*ny))
    for ax in fig.axes: ax.axis('off')
    for i in range(nb_features):
        plt.sca(axes.ravel()[i])
        plt.imshow(features[:,:,i], cmap=cmap)
        #plt.axis('off')
        if (labels is None):
            plt.title('Feature {}'.format(i), fontsize=6)
        else:
            plt.title(labels[i], fontsize=6)
    return fig

def force_integer_ticks(axes):
    """
    Parameters
    ----------
    
    axes : `Axes` on which to apply the new ticks
    """
    axes.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1,2,5,10]))
    axes.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[1,2,5,10])) 
    