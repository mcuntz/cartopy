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


def add_cyclic_point(data, coord=None, rowcoord=None, axis=-1):
    """
    Add a cyclic point to an array and optionally a corresponding
    coordinate.

    Parameters
    ----------
    data : ndarray
        An n-dimensional array of data to add a cyclic point to.
    coord: ndarray, optional
        A 1- or 2-dimensional array which specifies the coordinate values for
        the dimension the cyclic point is to be added to. Defaults to None.

        If `coord` is given than add_cyclic_point checks if cyclic point is
        already present by checking `sin(coord[0]) == sin(coord[-1])`.
        No point is added if cyclic point was detected.

        Length of `coord` must be `data.shape[axis]` if 1-dimensional.

        `coord.shape[-1]` must be `data.shape[axis]` if 2-dimensional.
    rowcoord: ndarray, optional
        A 2-dimensional array with the variable of the row coordinate.
        The cyclic point simply copies the last column. Only considered if
        `coord` is given and is 2-dimensional. Defaults to None.

        `rowcoord.shape[-1]` must be `data.shape[axis]`.
    axis: optional
        Specifies the axis of the data array to add the cyclic point to.
        Defaults to the right-most axis.

    Returns
    -------
    cyclic_data
        The data array with a cyclic point added.
    cyclic_coord
        The coordinate with a cyclic point, only returned if the `coord`
        keyword was supplied.
    cyclic_rowcoord
        The row coordinate with the last column duplicated, only returned
        if `coord` was 2-dimensional and the `lat` keyword was supplied.

    Examples
    --------
    Adding a cyclic point to a data array, where the cyclic dimension is
    the right-most dimension.
    >>> import numpy as np
    >>> np.set_printoptions(legacy='1.13')
    >>> data = np.ones([5, 6]) * np.arange(6)
    >>> cyclic_data = add_cyclic_point(data)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]

    Adding a cyclic point to a data array and an associated coordinate.
    >>> lons = np.arange(0, 360, 60)
    >>> cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lons)
    [   0.  60.  120.  180.  240.  300.  360.]

    Adding a cyclic point to a data array and an associated 2-dimensional
    coordinate.
    >>> lons = np.arange(0, 360, 60)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d = add_cyclic_point(data, coord=lon2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lon2d)
    [[   0.   60.  120.  180.  240.  300.  360.]
     [   0.   60.  120.  180.  240.  300.  360.]
     [   0.   60.  120.  180.  240.  300.  360.]
     [   0.   60.  120.  180.  240.  300.  360.]
     [   0.   60.  120.  180.  240.  300.  360.]]

    Adding a cyclic point to a data array and an associated 2-dimensional
    coordinate and a second raw variable.
    >>> lons = np.arange(0, 360, 60)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic_point(
    ...     data, coord=lon2d, rowcoord=lat2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]
     [ 0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lon2d)
    [[   0.   60.  120.  180.  240.  300.  360.]
     [   0.   60.  120.  180.  240.  300.  360.]
     [   0.   60.  120.  180.  240.  300.  360.]
     [   0.   60.  120.  180.  240.  300.  360.]
     [   0.   60.  120.  180.  240.  300.  360.]]
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
    >>> cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic_point(
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
    if coord is not None:
        if (coord.ndim < 1) or (coord.ndim > 2):
            estr  = 'The coordinate must be 1- or 2-dimensional.'
            estr += ' coord.shape: '+str(coord.shape)
            raise ValueError(estr)
        if (coord.ndim == 1):
            if len(coord) != data.shape[axis]:
                estr  = 'The length of the coordinate does not match'
                estr += ' the size of the corresponding dimension of'
                estr += ' the data array: len(coord) ='
                estr += ' {}, data.shape[{}] = {}.'.format(
                    len(coord), axis, data.shape[axis])
                raise ValueError(estr)
            # check if cyclic point already present
            # atol=1e-5 because coordinates often float32
            # and np.sin(np.deg2rad(np.float32(360.))) == 1.7484555e-07
            # add a bit of tolerance, e.g. cyclic points from rotated grid
            # I have seen differences of > 1e-5 in this case. Test is on
            # sine so that atol=1e-5 seems sufficient because 180/pi ~ 57.
            if np.ma.allclose(np.ma.sin(np.deg2rad(coord[0])),
                              np.ma.sin(np.deg2rad(coord[-1])),
                              atol=1.0e-5):
                if rowcoord is None:
                    return data, coord
                else:
                    return data, coord, rowcoord
            # # old code: must be equally spaced, adding diff
            # # delta_coord = np.diff(coord)
            # # if not np.allclose(delta_coord, delta_coord[0]):
            # #     raise ValueError('The coordinate must be equally spaced.')
            # # new_coord = ma.concatenate((coord,
            # #                             coord[-1:] + delta_coord[0]))
            # new code: just add 360 degree to first lon
            new_coord = np.ma.concatenate((coord, coord[0:1] + 360.))
        if (coord.ndim == 2):
            if coord.shape[-1] != data.shape[axis]:
                estr  = 'coord.shape[-1] does not match'
                estr += ' the size of the corresponding dimension of'
                estr += ' the data array: coord.shape[-1] ='
                estr += ' {}, data.shape[{}] = {}.'.format(
                    coord.shape[-1], axis, data.shape[axis])
                raise ValueError(estr)
            if rowcoord is not None:
                if not np.all(coord.shape == rowcoord.shape):
                    estr  = 'rowcoord.shape does not match'
                    estr += ' coord.shape: coord.shape[] =,'
                    estr += ' {}, rowcoord.shape = {}.'.format(
                        coord.shape, rowcoord.shape)
                    raise ValueError(estr)
            # check if cyclic point already present
            # atol=1e-5 see comment above
            if np.ma.allclose(np.ma.sin(np.deg2rad(coord[:, 0])),
                              np.ma.sin(np.deg2rad(coord[:, -1])),
                              atol=1.0e-5):
                if rowcoord is None:
                    return data, coord
                else:
                    return data, coord, rowcoord
            new_coord = np.ma.append(coord, coord[:, 0:1] + 360., axis=1)
            if rowcoord is not None:
                new_rowcoord = np.ma.append(rowcoord, rowcoord[:, -1:], axis=1)
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        estr = 'The specified axis does not correspond to an array dimension.'
        raise ValueError(estr)
    new_data = np.ma.concatenate((data, data[tuple(slicer)]), axis=axis)
    if coord is None:
        return new_data
    else:
        if (coord.ndim == 2) and (rowcoord is not None):
            return new_data, new_coord, new_rowcoord
        else:
            return new_data, new_coord
