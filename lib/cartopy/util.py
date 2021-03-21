# Copyright Cartopy Contributors
#
# This file is part of Cartopy and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
This module contains utilities that are useful in conjunction with
cartopy.

"""
import numpy as np
import numpy.ma as ma


def add_cyclic_point(data, coord=None, axis=-1):
    """
    Add a cyclic point to an array and optionally a corresponding
    coordinate.

    Parameters
    ----------
    data
        An n-dimensional array of data to add a cyclic point to.
    coord : optional
        A 1-dimensional array which specifies the coordinate values for
        the dimension the cyclic point is to be added to. The coordinate
        values must be regularly spaced. Defaults to None.
    axis : optional
        Specifies the axis of the data array to add the cyclic point to.
        Defaults to the right-most axis.

    Returns
    -------
    cyclic_data
        The data array with a cyclic point added.
    cyclic_coord
        The coordinate with a cyclic point, only returned if the coord
        keyword was supplied.

    Examples
    --------
    Adding a cyclic point to a data array, where the cyclic dimension is
    the right-most dimension.

    >>> import numpy as np
    >>> data = np.ones([5, 6]) * np.arange(6)
    >>> cyclic_data = add_cyclic_point(data)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]

    Adding a cyclic point to a data array and an associated coordinate

    >>> lons = np.arange(0, 360, 60)
    >>> cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lons)
    [  0  60 120 180 240 300 360]

    """
    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                                 len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value


def _add_cyclic_data(data, axis=-1):
    """
    Add a cyclic point to a data array.

    Parameters
    ----------
    data : ndarray
        An n-dimensional array of data to add a cyclic point to.
    axis : int, optional
        Specifies the axis of the data array to add the cyclic point to.
        Defaults to the right-most axis.

    Returns
    -------
    The data array with a cyclic point added.
    """
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        estr = 'The specified axis does not correspond to an array dimension.'
        raise ValueError(estr)
    npc = np.ma if np.ma.is_masked(data) else np
    return npc.concatenate((data, data[tuple(slicer)]), axis=axis)


def _add_cyclic_lon(lon, axis=-1, cyclic=360):
    """
    Add a cyclic point to a longitude array.

    Parameters
    ----------
    lon : ndarray
        An array which specifies the coordinate values for
        the dimension the cyclic point is to be added to.
    axis : int, optional
        Specifies the axis of the longitude array to add the cyclic point to.
        Defaults to the right-most axis.
    cyclic : float, optional
        Width of periodic domain (default: 360)

    Returns
    -------
    The coordinate array `lon` with a cyclic point added.
    """
    npc = np.ma if np.ma.is_masked(lon) else np
    # get cyclic longitudes
    # clon is the code from basemap (addcyclic)
    # https://github.com/matplotlib/basemap/blob/master/lib/mpl_toolkits/basemap/__init__.py
    clon = (np.take(lon, [0], axis=axis) +
            cyclic * np.sign(np.diff(np.take(lon, [0, -1], axis=axis),
                                     axis=axis)))
    # basemap ensures that the values do not exceed cyclic
    # (next code line). We do not do this to deal with rotated grids that
    # might have values not exactly 0.
    #     clon = npc.where(clon <= cyclic, clon, np.mod(clon, cyclic))
    return npc.concatenate((lon, clon), axis=axis)


def _has_cyclic(lon, axis=-1, cyclic=360, precision=1e-4):
    """
    Check if longitudes already have a cyclic point.

    Checks all differences between the first and last
    longitudes along `axis` to be less than `precision`.

    Parameters
    ----------
    lon : ndarray
        An array with the coordinate values to be checked for cyclic points.
    axis : int, optional
        Specifies the axis of the `lon` array to be checked.
        Defaults to the right-most axis.
    cyclic : float, optional
        Width of periodic domain (default: 360).
    precision : float, optional
        Maximal difference between first and last longitude to detect
        cyclic point (default: 1e-4).

    Returns
    -------
    True if a cyclic point was detected along the given axis,
    False otherwise.
    """
    npc = np.ma if np.ma.is_masked(lon) else np
    # transform to 0-cyclic, assuming e.g. -180 to 180 if any < 0
    lon1 = np.mod(npc.where(lon < 0, lon + cyclic, lon), cyclic)
    dd = np.diff(np.take(lon1, [0, -1], axis=axis), axis=axis)
    if npc.all(np.abs(dd) < precision):
        return True
    else:
        return False


def add_cyclic(data, coord=None, rowcoord=None, axis=-1,
               cyclic=360, precision=1e-4):
    """
    Add a cyclic point to an array and optionally corresponding
    column (`coord` ~ longitudes) and row coordinates
    (`rowcoord` ~ latitudes).

    The call is `add_cyclic(data[, coord[, rowcoord]])`.

    Checks all differences between the first and last
    coordinates along `axis` to be less than `precision`.

    Parameters
    ----------
    data : ndarray
        An n-dimensional array of data to add a cyclic point to.
    coord : ndarray, optional
        An n-dimensional array which specifies the coordinate values
        for the dimension the cyclic point is to be added to, i.e. normally the
        longitudes. Defaults to None.

        If `coord` is given then *add_cyclic* checks if a cyclic point is
        already present by checking all differences between the first and last
        coordinates to be less than `precision`.
        No point is added if a cyclic point was detected.

        If coord is 1- or 2-dimensional, `coord.shape[-1]` must equal
        `data.shape[axis]`, otherwise `coord.shape[axis]` must equal
        `data.shape[axis]`.
    rowcoord : ndarray, optional
        An n-dimensional array with the variable of the row
        coordinate, i.e. normally the latitudes.
        The cyclic point simply copies the last column. Defaults to None.

        `rowcoord.shape[-1]` must be `data.shape[axis]` if rowcoord is
         1- or 2-dimensional,
        `rowcoord.shape[axis]` must be `data.shape[axis]` otherwise.
    axis : int, optional
        Specifies the axis of the arrays to add the cyclic point to,
        i.e. axis with changing longitudes. Defaults to the right-most axis.
    cyclic : int or float, optional
        Width of periodic domain (default: 360).
    precision : float, optional
        Maximal difference between first and last coordinate to detect
        cyclic point (default: 1e-4).

    Returns
    -------
    cyclic_data
        The data array with a cyclic point added.
    cyclic_coord
        The coordinate with a cyclic point, only returned if the `coord`
        keyword was supplied.
    cyclic_rowcoord
        The row coordinate with the last column of the cyclic axis duplicated,
        only returned if `coord` was 2- or n-dimensional and the `rowcoord`
        keyword was supplied.

    Examples
    --------
    Adding a cyclic point to a data array, where the cyclic dimension is
    the right-most dimension.
    .. testsetup::
        >>> from distutils.version import LooseVersion
        >>> import numpy as np
        >>> if LooseVersion(np.__version__) >= '1.14.0':
        ...     # To provide consistent doctests.
        ...     np.set_printoptions(legacy='1.13')
    >>> import numpy as np
    >>> data = np.ones([5, 6]) * np.arange(6)
    >>> cyclic_data = add_cyclic(data)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]

    Adding a cyclic point to a data array and an associated coordinate.
    >>> lons = np.arange(0, 360, 60)
    >>> cyclic_data, cyclic_lons = add_cyclic(data, coord=lons)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lons)
    [  0  60 120 180 240 300 360]

    Adding a cyclic point to a data array and an associated 2-dimensional
    coordinate.
    >>> lons = np.arange(0, 360, 60)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d = add_cyclic(data, coord=lon2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lon2d)
    [[  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]]

    Adding a cyclic point to a data array and the associated 2-dimensional
    x and y coordinates.
    >>> lons = np.arange(0, 360, 60)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(
    ...     data, coord=lon2d, rowcoord=lat2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lon2d)
    [[  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]]
    >>> print(cyclic_lat2d)
    [[-90. -90. -90. -90. -90. -90. -90.]
     [-54. -54. -54. -54. -54. -54. -54.]
     [-18. -18. -18. -18. -18. -18. -18.]
     [ 18.  18.  18.  18.  18.  18.  18.]
     [ 54.  54.  54.  54.  54.  54.  54.]]

    Not adding a cyclic point if cyclic point detected in coord.
    >>> lons = np.arange(0, 361, 72)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(
    ...     data, coord=lon2d, rowcoord=lat2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5.]
     [ 0. 1. 2. 3. 4. 5.]
     [ 0. 1. 2. 3. 4. 5.]
     [ 0. 1. 2. 3. 4. 5.]
     [ 0. 1. 2. 3. 4. 5.]]
    >>> print(cyclic_lon2d)
    [[  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]]
    >>> print(cyclic_lat2d)
    [[-90. -90. -90. -90. -90. -90.]
     [-54. -54. -54. -54. -54. -54.]
     [-18. -18. -18. -18. -18. -18.]
     [ 18.  18.  18.  18.  18.  18.]
     [ 54.  54.  54.  54.  54.  54.]]
    """
    if coord is None:
        return _add_cyclic_data(data, axis=axis)
    # if coord
    if (coord.ndim > 2):
        caxis = axis
    else:
        caxis = -1
    if coord.shape[caxis] != data.shape[axis]:
        estr = (f'coord.shape[{caxis}] does not match the size of the'
                f' corresponding dimension of the data array:'
                f' coord.shape[{caxis}] = {coord.shape[caxis]},'
                f' data.shape[{axis}] = {data.shape[axis]}.')
        raise ValueError(estr)
    if _has_cyclic(coord, axis=caxis, cyclic=cyclic, precision=precision):
        if rowcoord is None:
            return data, coord
        # if rowcoord
        return data, coord, rowcoord
    # if not _has_cyclic, add cyclic points to data and coord
    odata = _add_cyclic_data(data, axis=axis)
    ocoord = _add_cyclic_lon(coord, axis=caxis, cyclic=cyclic)
    if rowcoord is None:
        return odata, ocoord
    # if rowcoord
    if (rowcoord.ndim > 2):
        raxis = axis
    else:
        raxis = -1
    if rowcoord.shape[raxis] != data.shape[axis]:
        estr = (f'rowcoord.shape[{raxis}] does not match the size of the'
                f' corresponding dimension of the data array:'
                f' rowcoord.shape[{raxis}] = {rowcoord.shape[raxis]},'
                f' data.shape[{axis}] = {data.shape[axis]}.')
        raise ValueError(estr)
    orowcoord = _add_cyclic_data(rowcoord, axis=raxis)
    return odata, ocoord, orowcoord
