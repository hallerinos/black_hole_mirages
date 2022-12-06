import tinyarray as ta
import numpy as np
from kwant import _common, builder, system
from kwant._common import deprecate_args

_p = _common.lazy_import('_plotter')

def _make_figure(dpi, fig_size, use_pyplot=False):
    from matplotlib import pyplot
    fig = pyplot.figure()
    return fig

def _gamma_compress(linear):
    """Compress linear sRGB into sRGB."""
    if linear <= 0.0031308:
        return 12.92 * linear
    else:
        a = 0.055
        return (1 + a) * linear ** (1 / 2.4) - a

_gamma_compress = np.vectorize(_gamma_compress, otypes=[float])

def _gamma_expand(corrected):
    """Expand sRGB into linear sRGB."""
    if corrected <= 0.04045:
        return corrected / 12.92
    else:
        a = 0.055
        return ((corrected + a) / (1 + a))**2.4

_gamma_expand = np.vectorize(_gamma_expand, otypes=[float])

def _linear_cmap(a, b):
    """Make a colormap that linearly interpolates between the colors a and b."""
    a = _p.matplotlib.colors.colorConverter.to_rgb(a)
    b = _p.matplotlib.colors.colorConverter.to_rgb(b)
    a_linear = _gamma_expand(a)
    b_linear = _gamma_expand(b)
    color_diff = a_linear - b_linear
    palette = (np.linspace(0, 1, 256).reshape((-1, 1))
               * color_diff.reshape((1, -1)))
    palette += b_linear
    palette = _gamma_compress(palette)
    return _p.matplotlib.colors.ListedColormap(palette)

# Determine the optimal bump function width from the absolute and
# relative widths provided, and the lengths of all the hoppings in the system
def _optimal_width(lens, abswidth, relwidth, bbox_size):
    if abswidth is None:
        if relwidth is None:
            unique_lens = np.unique(lens)
            longest = unique_lens[-1]
            for shortest_nonzero in unique_lens:
                if shortest_nonzero / longest > 1e-3:
                    break
            width = 4 * shortest_nonzero
        else:
            width = relwidth * np.max(bbox_size)
    else:
        width = abswidth

    return width

# Create empty field array that covers the bounding box plus
# some additional padding
def _create_field(dim, bbox_size, width, n, is_current):
    field_shape = np.zeros(dim + 1, int)
    field_shape[dim] = dim if is_current else 1
    for d in range(dim):
        field_shape[d] = int(bbox_size[d] * n / width + n)
        if field_shape[d] % 2:
            field_shape[d] += 1
    field = np.zeros(field_shape)
    # padding is width / 2
    return field, width / 2

# We generate the smoothing function by convolving the current
# defined on a line between the two sites with
# f(ρ, z) = (1 - ρ^2 - z^2)^2 Θ(1 - ρ^2 - z^2), where ρ and z are
# cylindrical coords defined with respect to the hopping.
# 'F' is the result of the convolution.
def _smoothing(rho, z):
    r = 1 - rho * rho
    r[r < 0] = 0
    r = np.sqrt(r)
    m = np.clip(z, -r, r)
    rr = r * r
    rrrr = rr * rr
    mm = m * m
    return m * (mm * (mm/5 - (2/3) * rr) + rrrr) + (8 / 15) * rrrr * r


def current_kernel(coords, direction, length):
    z = np.dot(coords, direction)
    rho = np.sqrt(np.abs(np.sum(coords * coords) - z * z))
    magn = (_smoothing(rho, z) - _smoothing(rho, z - length))
    return direction * magn[..., None]

# We need to normalize the smoothing function so that it has unit cross
# section in the plane perpendicular to the hopping. This is equivalent
# to normalizing the integral of 'f' over the unit hypersphere to 1.
# The smoothing function goes as F(ρ) = (16/15) (1 - ρ^2)^(5/2) in the
# plane perpendicular to the hopping, so the cross section is:
# A_n = (16 / 15) * σ_n * ∫_0^1 ρ^(n-1) (1 - ρ^2)^(5/2) dρ
# where σ_n is the surface element prefactor (2 in 2D, 2π in 3D). Rather
# that calculate A_n every time, we hard code its value for 1, 2 and 3D.
_smoothing_cross_sections = [16 / 15, np.pi / 3, 32 * np.pi / 105]

# interpolate a discrete scalar or vector field.
def _interpolate_field(dim, elements, discrete_field, bbox, width,
                       padding, field_out):

    field_shape = np.array(field_out.shape)
    bbox_min, bbox_max = bbox

    scale = 2 / width

    # if density elements is shape (nsites, dim)
    # if current elements is shape (nhops, 2, dim)
    assert elements.shape[-1] == dim
    is_current = len(elements.shape) == 3
    if is_current:
        assert elements.shape[1] == 2
        dirs = elements[:, 1] - elements[:, 0]
        lens = np.sqrt(np.sum(dirs * dirs, axis=-1))
        dirs /= lens[:, None]
        lens = lens * scale

    if is_current:
        pos_offsets = elements[:, 0]  # first site in hopping
        kernel = current_kernel
    else:
        pos_offsets = elements  # sites themselves
        kernel = density_kernel

    region = [np.linspace(bbox_min[d] - padding,
                          bbox_max[d] + padding,
                          field_shape[d])
              for d in range(dim)]

    grid_density = (field_shape[:dim] - 1) / (bbox_max + 2*padding - bbox_min)

    # slices for indexing 'field' and 'region' array
    slices = np.empty((len(discrete_field), dim, 2), int)
    if is_current:
        mn = np.min(elements, 1)
        mx = np.max(elements, 1)
    else:
        mn = mx = elements
    slices[:, :, 0] = np.floor((mn - bbox_min) * grid_density)
    slices[:, :, 1] = np.ceil((mx + 2*padding - bbox_min) * grid_density)

    for i in range(len(discrete_field)):

        if not np.diff(slices[i]).all() or not discrete_field[i]:
            # Zero volume or zero field: nothing to do.
            continue

        field_slice = tuple([slice(*slices[i, d]) for d in range(dim)])

        # Coordinates of the grid points that are within range of the current
        # hopping.
        coords = np.array(
            np.meshgrid(
                *[region[d][field_slice[d]] for d in range(dim)],
                sparse=True, indexing='ij'
            ),
            dtype=object
        )

        # Convert "coords" into scaled distances from pos_offset
        coords -= pos_offsets[i]
        coords *= scale
        magns = kernel(coords, dirs[i], lens[i]) if is_current else kernel(coords)
        magns *= discrete_field[i]

        field_out[field_slice] += magns

    field_out *= scale / _smoothing_cross_sections[dim - 1]

def mystreamplot(field, box, cmap=None, bgcolor=None, linecolor='k',
               max_linewidth=3, min_linewidth=1, density=2/9,
               colorbar=True, file=None,
               show=True, dpi=None, fig_size=None, ax=None,
               vmax=None, **kwargs):
    fig = _streamplot_matplotlib(field, box, cmap, bgcolor, linecolor,
            max_linewidth, min_linewidth, density, colorbar, file,
            show, dpi, fig_size, ax, vmax, **kwargs)

def _maybe_output_fig(fig, file=None, show=True):
    """Output a matplotlib figure using a given output mode.

    Parameters
    ----------
    fig : matplotlib.figure.Figure instance
        The figure to be output.
    file : string or a file object
        The name of the target file or the target file itself
        (opened for writing).
    show : bool
        Whether to call ``matplotlib.pyplot.show()``.  Only has an effect if
        not saving to a file.

    Notes
    -----
    The behavior of this function producing a file is different from that of
    matplotlib in that the `dpi` attribute of the figure is used by defaul
    instead of the matplotlib config setting.
    """
    if fig is None:
        return

    if file is not None:
        fig.canvas.print_figure(file, dpi=fig.dpi)
    elif show:
        # If there was no file provided, pyplot should already be available
        # and we can import it safely without additional warnings.
        from matplotlib import pyplot
        pyplot.show()


def _streamplot_matplotlib(field, box, cmap, bgcolor, linecolor,
               max_linewidth, min_linewidth, density, colorbar, file,
               show, dpi, fig_size, ax, vmax, **kwargs):
    """Draw streamlines of a flow field in Kwant style

    Solid colored streamlines are drawn, superimposed on a color plot of
    the flow speed that may be disabled by setting `bgcolor`.  The width
    of the streamlines is proportional to the flow speed.  Lines that
    would be thinner than `min_linewidth` are blended in a perceptually
    correct way into the background color in order to create the
    illusion of arbitrarily thin lines.  (This is done because some plot
    engines like PDF do not support lines of arbitrarily thin width.)

    Internally, this routine uses matplotlib's streamplot.

    Parameters
    ----------
    field : 3d arraylike of float
        2d array of 2d vectors.
    box : 2-sequence of 2-sequences of float
        the extents of `field`: ((x0, x1), (y0, y1))
    cmap : colormap, optional
        Colormap for the background color plot.  When not set the colormap
        "kwant_red" is used by default, unless `bgcolor` is set.
    bgcolor : color definition, optional
        The solid color of the background.  Mutually exclusive with `cmap`.
    linecolor : color definition
        Color of the flow lines.
    max_linewidth : float
        Width of lines at maximum flow speed.
    min_linewidth : float
        Minimum width of lines before blending into the background color begins.
    density : float
        Number of flow lines per point of the field.  The default value
        of 2/9 is chosen to show two lines per default width of the
        interpolation bump of `~kwant.plotter.interpolate_current`.
    colorbar : bool
        Whether to show a colorbar if a colormap is used. Ignored if `ax` is
        provided.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether ``matplotlib.pyplot.show()`` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float or `None`
        Number of pixels per inch.  If not set the ``matplotlib`` default is
        used.
    fig_size : tuple or `None`
        Figure size `(width, height)` in inches.  If not set, the default
        ``matplotlib`` value is used.
    ax : ``matplotlib.axes.Axes`` instance or `None`
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ignored.
    vmax : float or `None`
        The upper saturation limit for the colormap; flows higher than
        this will saturate.  Note that there is no corresponding vmin
        option, vmin being fixed at zero.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.
    """

    # Matplotlib's "density" is in units of 30 streamlines...
    density *= 1 / 30 * ta.array(field.shape[:2], int)

    # Matplotlib plots images like matrices: image[y, x].  We use the opposite
    # convention: image[x, y].  Hence, it is necessary to transpose.
    field = field.transpose(1, 0, 2)

    if field.shape[-1] != 2 or field.ndim != 3:
        raise ValueError("Only 2D field can be plotted.")

    if bgcolor is None:
        if cmap is None:
            cmap = _p.kwant_red_matplotlib
        cmap = _p.matplotlib.cm.get_cmap(cmap)
        bgcolor = cmap(0)[:3]
    elif cmap is not None:
        raise ValueError("The parameters 'cmap' and 'bgcolor' are "
                         "mutually exclusive.")

    if ax is None:
        fig = _make_figure(dpi, fig_size, use_pyplot=(file is None))
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
    else:
        fig = None

    X = np.linspace(*box[0], num=field.shape[1])
    Y = np.linspace(*box[1], num=field.shape[0])

    speed = np.linalg.norm(field, axis=-1)
    if vmax is None:
        vmax = np.max(speed) or 1

    if cmap is None:
        ax.set_axis_bgcolor(bgcolor)
    else:
        image = ax.imshow(speed, cmap=cmap,
                          interpolation='bicubic',
                          extent=[e for c in box for e in c],
                          origin='lower', vmin=0, vmax=vmax)

    linewidth = max_linewidth / vmax * speed
    color = linewidth / min_linewidth
    thin = linewidth < min_linewidth
    linewidth[thin] = min_linewidth
    color[~ thin] = 1

    line_cmap = _linear_cmap(linecolor, bgcolor)

    ax.streamplot(X, Y, field[:,:,0], field[:,:,1],
                  density=density, linewidth=linewidth,
                  color=color, cmap=line_cmap, arrowstyle='->',
                  norm=_p.matplotlib.colors.Normalize(0, 1), **kwargs)

    ax.set_xlim(*box[0])
    ax.set_ylim(*box[1])

    if colorbar and cmap and fig is not None:
        fig.colorbar(image)

    _maybe_output_fig(fig, file=file, show=show)

    return fig

def interpolate_current(syst, current, relwidth=None, abswidth=None, n=9):
    """Interpolate currents in a system onto a regular grid.

    The system graph together with current intensities defines a "discrete"
    current density field where the current density is non-zero only on the
    straight lines that connect sites that are coupled by a hopping term.

    To make this vector field easier to visualize and interpret at different
    length scales, it is smoothed by convoluting it with the bell-shaped bump
    function ``f(r) = max(1 - (2*r / width)**2, 0)**2``.  The bump width is
    determined by the `relwidth` and `abswidth` parameters.

    This routine samples the smoothed field on a regular (square or cubic)
    grid.

    Parameters
    ----------
    syst : A finalized system
        The system on which we are going to calculate the field.
    current : '1D array of float'
        Must contain the intensity on each hoppings in the same order that they
        appear in syst.graph.
    relwidth : float or `None`
        Relative width of the bumps used to generate the field, as a fraction
        of the length of the longest side of the bounding box.  This argument
        is only used if `abswidth` is not given.
    abswidth : float or `None`
        Absolute width of the bumps used to generate the field.  Takes
        precedence over `relwidth`.  If neither is given, the bump width is set
        to four times the length of the shortest hopping.
    n : int
        Number of points the grid must have over the width of the bump.

    Returns
    -------
    field : n-d arraylike of float
        n-d array of n-d vectors.
    box : sequence of 2-sequences of float
        the extents of `field`: ((x0, x1), (y0, y1), ...)

    """

    if len(current) != syst.graph.num_edges:
        raise ValueError("Current and hoppings arrays do not have the same"
                         " length.")

    # hops: hoppings (pairs of points)
    dim = len(syst.sites[0].pos)
    hops = np.empty((syst.graph.num_edges // 2, 2, dim))
    # Take the average of the current flowing each way along the hoppings
    current_one_way = np.empty(syst.graph.num_edges // 2)
    seen_hoppings = dict()
    kprime = 0
    for k, (i, j) in enumerate(syst.graph):
        if (j, i) in seen_hoppings:
            current_one_way[seen_hoppings[j, i]] -= current[k]
        else:
            current_one_way[kprime] = current[k]
            hops[kprime][0] = syst.sites[j].pos
            hops[kprime][1] = syst.sites[i].pos
            seen_hoppings[i, j] = kprime
            kprime += 1
    current = current_one_way / 2

    min_hops = np.min(hops, 1)
    max_hops = np.max(hops, 1)
    bbox_min = np.min(min_hops, 0)
    bbox_max = np.max(max_hops, 0)
    bbox_size = bbox_max - bbox_min

    # lens: scaled lengths of hoppings
    # dirs: normalized directions of hoppings
    dirs = hops[:, 1] - hops[:, 0]
    lens = np.sqrt(np.sum(dirs * dirs, -1))
    dirs /= lens[:, None]
    width = _optimal_width(lens, abswidth, relwidth, bbox_size)


    field, padding = _create_field(dim, bbox_size, width, n, is_current=True)
    boundaries = tuple((bbox_min[d] - padding, bbox_max[d] + padding)
                        for d in range(dim))
    _interpolate_field(dim, hops, current,
                       (bbox_min, bbox_max), width, padding, field)

    return field, boundaries

def mycurrent(syst, current, relwidth=0.05, **kwargs):
    """Show an interpolated current defined for the hoppings of a system.

    The system graph together with current intensities defines a "discrete"
    current density field where the current density is non-zero only on the
    straight lines that connect sites that are coupled by a hopping term.

    To make this scalar field easier to visualize and interpret at different
    length scales, it is smoothed by convoluting it with the bell-shaped bump
    function ``f(r) = max(1 - (2*r / width)**2, 0)**2``.  The bump width is
    determined by the ``relwidth`` parameter.

    This routine samples the smoothed field on a regular (square or cubic) grid
    and displays it using an enhanced variant of matplotlib's streamplot.

    This is a convenience function that is equivalent to
    ``streamplot(*interpolate_current(syst, current, relwidth), **kwargs)``.
    The longer form makes it possible to tweak additional options of
    `~kwant.plotter.interpolate_current`.

    Parameters
    ----------
    syst : `kwant.system.FiniteSystem`
        The system for which to plot the ``current``.
    current : sequence of float
        Sequence of values defining currents on each hopping of the system.
        Ordered in the same way as ``syst.graph``. This typically will be
        the result of evaluating a `~kwant.operator.Current` operator.
    relwidth : float or `None`
        Relative width of the bumps used to smooth the field, as a fraction
        of the length of the longest side of the bounding box.
    **kwargs : various
        Keyword args to be passed verbatim to `kwant.plotter.streamplot`.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if ``ax`` is not set, else None.

    See Also
    --------
    kwant.plotter.density
    """
    with _common.reraise_warnings(4):
        return mystreamplot(*interpolate_current(syst, current, relwidth),
                          **kwargs)
