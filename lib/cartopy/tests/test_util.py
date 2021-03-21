# Copyright Cartopy Contributors
#
# This file is part of Cartopy and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal, assert_equal
import pytest

from cartopy.util import add_cyclic_point
from cartopy.util import add_cyclic


class Test_add_cyclic_point:

    @classmethod
    def setup_class(cls):
        cls.lons = np.arange(0, 360, 60)
        cls.data2d = np.ones([3, 6]) * np.arange(6)
        cls.data4d = np.ones([4, 6, 2, 3]) * \
            np.arange(6)[..., np.newaxis, np.newaxis]

    def test_data_only(self):
        c_data = add_cyclic_point(self.data2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_and_coord(self):
        c_data, c_lons = add_cyclic_point(self.data2d, coord=self.lons)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_only_with_axis(self):
        c_data = add_cyclic_point(self.data4d, axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_and_coord_with_axis(self):
        c_data, c_lons = add_cyclic_point(self.data4d, coord=self.lons, axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_masked_data(self):
        new_data = ma.masked_less(self.data2d, 3)
        c_data = add_cyclic_point(new_data)
        r_data = ma.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_invalid_coord_dimensionality(self):
        lons2d = np.repeat(self.lons[np.newaxis], 3, axis=0)
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic_point(self.data2d, coord=lons2d)

    def test_invalid_coord_size(self):
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic_point(self.data2d,
                                              coord=self.lons[:-1])

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            add_cyclic_point(self.data2d, axis=-3)


class Test_add_cyclic:
    """
    Test def add_cyclic(data, coord=None, rowcoord=None, axis=-1,
                        cyclic=360, precision=1e-4):
    - variations of data, coord, and rowcoord with and without axis keyword
    - different unit of coord - cyclic
    - detection of cyclic points - precision
    - error catching
    """

    @classmethod
    def setup_class(cls):
        # 2d and 4d data
        cls.data2d = np.ones([3, 6]) * np.arange(6)
        cls.data4d = np.ones([4, 6, 2, 3]) * \
            np.arange(6)[..., np.newaxis, np.newaxis]
        # 1d lat (5) and lon (6)
        # len(lat) != data.shape[0]
        # len(lon) == data.shape[1]
        cls.lons = np.arange(0, 360, 60)
        cls.lats = np.arange(-90, 90, 180/5)
        # 2d lat and lon
        cls.lon2d, cls.lat2d = np.meshgrid(cls.lons, cls.lats)
        # 3d lat and lon but with different 3rd dimension (4) as 4d data (2)
        cls.lon3d = np.repeat(cls.lon2d, 4).reshape((*cls.lon2d.shape, 4))
        cls.lat3d = np.repeat(cls.lat2d, 4).reshape((*cls.lat2d.shape, 4))

    def test_data_only(self):
        '''Test only data no coord given'''
        c_data = add_cyclic(self.data2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_only_ignore_rowcoord(self):
        '''Test rowcoord given but no coord'''
        c_data = add_cyclic(self.data2d, rowcoord=self.lat2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_and_coord_1d(self):
        '''Test data 2d and coord 1d'''
        c_data, c_lons = add_cyclic(self.data2d, coord=self.lons)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_2d(self):
        '''Test data and coord 2d'''
        c_data, c_lons = add_cyclic(self.data2d, coord=self.lon2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_1d_rowcoord_2d(self):
        '''Test data and rowcoord 2d and coord 1d'''
        c_data, c_lons, c_lats = add_cyclic(self.data2d, coord=self.lons,
                                            rowcoord=self.lat2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        r_lats = np.concatenate((self.lat2d, self.lat2d[:, -1:]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_data_and_coord_rowcoord_2d(self):
        '''Test data, coord, and rowcoord 2d'''
        c_data, c_lons, c_lats = add_cyclic(self.data2d,
                                            coord=self.lon2d,
                                            rowcoord=self.lat2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        r_lats = np.concatenate((self.lat2d, self.lat2d[:, -1:]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_has_cyclic_1d(self):
        '''Test detection of cyclic point 1d'''
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        c_data, c_lons = add_cyclic(r_data, coord=r_lons)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_has_cyclic_2d(self):
        '''Test detection of cyclic point 2d'''
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        c_data, c_lons = add_cyclic(r_data, coord=r_lons)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_has_cyclic_2d_full(self):
        '''Test detection of cyclic point 2d including rowcoord'''
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        r_lats = np.concatenate((self.lat2d, self.lat2d[:, -1:]), axis=1)
        c_data, c_lons, c_lats = add_cyclic(r_data, coord=r_lons,
                                            rowcoord=r_lats)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_data_only_with_axis(self):
        '''Test axis keyword data only'''
        c_data = add_cyclic(self.data4d, axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_and_coord_with_axis_1d(self):
        '''Test axis keyword data 4d, coord 1d'''
        c_data, c_lons = add_cyclic(self.data4d, coord=self.lons, axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_with_axis_2d(self):
        '''Test axis keyword data 4d, coord 2d'''
        c_data, c_lons = add_cyclic(self.data4d, coord=self.lon2d,
                                    axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_with_axis_3d(self):
        '''Test axis keyword data 4d, coord 3d'''
        c_data, c_lons = add_cyclic(self.data4d, coord=self.lon3d,
                                    axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate(
            (self.lon3d,
             np.full((self.lon3d.shape[0], 1, self.lon3d.shape[2]), 360.)),
            axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_rowcoord_with_axis_2d(self):
        '''Test axis keyword data 4d, coord and rowcoord 2d'''
        c_data, c_lons, c_lats = add_cyclic(self.data4d,
                                            coord=self.lon2d,
                                            rowcoord=self.lat2d,
                                            axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        r_lats = np.concatenate((self.lat2d, self.lat2d[:, -1:]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_data_and_coord_rowcoord_with_axis_3d(self):
        '''Test axis keyword data 4d, coord and rowcoord 3d'''
        c_data, c_lons, c_lats = add_cyclic(self.data4d,
                                            coord=self.lon3d,
                                            rowcoord=self.lat3d,
                                            axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate(
            (self.lon3d,
             np.full((self.lon3d.shape[0], 1, self.lon3d.shape[2]), 360.)),
            axis=1)
        r_lats = np.concatenate((self.lat3d, self.lat3d[:, -1:, :]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_data_and_coord_rowcoord_with_axis_nd(self):
        '''Test axis keyword data 4d, coord 3d and rowcoord 2d'''
        c_data, c_lons, c_lats = add_cyclic(self.data4d,
                                            coord=self.lon3d,
                                            rowcoord=self.lat2d,
                                            axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate(
            (self.lon3d,
             np.full((self.lon3d.shape[0], 1, self.lon3d.shape[2]), 360.)),
            axis=1)
        r_lats = np.concatenate((self.lat2d, self.lat2d[:, -1:]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_masked_data(self):
        '''Test masked data'''
        new_data = ma.masked_less(self.data2d, 3)
        c_data = add_cyclic(new_data)
        r_data = ma.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_equal(ma.is_masked(c_data), True)

    def test_masked_data_and_coord_rowcoord_2d(self):
        '''Test masked data and coord'''
        new_data = ma.masked_less(self.data2d, 3)
        new_lon = ma.masked_less(self.lon2d, 2)
        c_data, c_lons, c_lats = add_cyclic(new_data,
                                            coord=new_lon,
                                            rowcoord=self.lat2d)
        r_data = ma.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        r_lats = np.concatenate((self.lat2d, self.lat2d[:, -1:]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)
        assert_equal(ma.is_masked(c_data), True)
        assert_equal(ma.is_masked(c_lons), True)
        assert_equal(ma.is_masked(c_lats), False)

    def test_cyclic(self):
        '''Test cyclic keyword with axis data 4d, coord 3d and rowcoord 2d'''
        new_lons = np.deg2rad(self.lon3d)
        new_lats = np.deg2rad(self.lat2d)
        c_data, c_lons, c_lats = add_cyclic(self.data4d, coord=new_lons,
                                            rowcoord=new_lats, axis=1,
                                            cyclic=np.deg2rad(360.))
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate(
            (new_lons,
             np.full((new_lons.shape[0], 1, new_lons.shape[2]),
                     np.deg2rad(360.))),
            axis=1)
        r_lats = np.concatenate((new_lats, new_lats[:, -1:]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_cyclic_has_cyclic(self):
        '''Test detection of cyclic point with cyclic keyword'''
        new_lons = np.deg2rad(self.lon2d)
        new_lats = np.deg2rad(self.lat2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate(
            (new_lons,
             np.full((new_lons.shape[0], 1), np.deg2rad(360.))),
            axis=1)
        r_lats = np.concatenate((new_lats, new_lats[:, -1:]), axis=1)
        c_data, c_lons, c_lats = add_cyclic(r_data, coord=r_lons,
                                            rowcoord=r_lats,
                                            cyclic=np.deg2rad(360.))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_precision_has_cyclic(self):
        '''Test precision keyword detecting cyclic point'''
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.+1e-3])))
        c_data, c_lons = add_cyclic(r_data, coord=r_lons, precision=1e-2)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_precision_has_cyclic_no(self):
        '''Test precision keyword detecting no cyclic point'''
        new_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        new_lons = np.concatenate((self.lons, np.array([360.+1e-3])))
        c_data, c_lons = add_cyclic(new_data, coord=new_lons, precision=2e-4)
        r_data = np.concatenate((new_data, new_data[:, :1]), axis=1)
        r_lons = np.concatenate((new_lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_invalid_coord_dimensionality(self):
        '''Catch wrong coord dimensions'''
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic(self.data2d, coord=self.lon3d)

    def test_invalid_rowcoord_dimensionality(self):
        '''Catch wrong rowcoord dimensions'''
        with pytest.raises(ValueError):
            c_data, c_lons, c_lats = add_cyclic(self.data2d,
                                                coord=self.lon2d,
                                                rowcoord=self.lat3d)

    def test_invalid_coord_size_1d(self):
        '''Catch wrong coord size 1d'''
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic(self.data2d,
                                        coord=self.lons[:-1])

    def test_invalid_coord_size_2d(self):
        '''Catch wrong coord size 2d'''
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic(self.data2d,
                                        coord=self.lon2d[:, :-1])

    def test_invalid_coord_size_3d(self):
        '''Catch wrong coord size 3d'''
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic(self.data4d,
                                        coord=self.lon3d[:, :-1, :], axis=1)

    def test_invalid_rowcoord_size(self):
        '''Catch wrong rowcoord size 2d'''
        with pytest.raises(ValueError):
            c_data, c_lons, c_lats = add_cyclic(
                self.data2d, coord=self.lon2d, rowcoord=self.lat2d[:, 1:])

    def test_invalid_axis(self):
        '''Catch wrong axis keyword'''
        with pytest.raises(ValueError):
            add_cyclic(self.data2d, axis=-3)
