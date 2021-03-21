# Copyright Cartopy Contributors
#
# This file is part of Cartopy and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest

from cartopy.util import add_cyclic_point


class Test_add_cyclic_point:

    @classmethod
    def setup_class(cls):
        cls.lons = np.arange(0, 360, 60)
        cls.lats = np.arange(-90, 90, 180/5)
        cls.lon2d, cls.lat2d = np.meshgrid(cls.lons, cls.lats)
        cls.data2d = np.ones([3, 6]) * np.arange(6)
        cls.data4d = np.ones([4, 6, 2, 3]) * \
            np.arange(6)[..., np.newaxis, np.newaxis]

    def test_data_only(self):
        c_data = add_cyclic_point(self.data2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_only_ignore_rowcoord(self):
        c_data = add_cyclic_point(self.data2d, rowcoord=self.lat2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_and_coord_1d(self):
        c_data, c_lons = add_cyclic_point(self.data2d, coord=self.lons)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_2d(self):
        c_data, c_lons = add_cyclic_point(self.data2d, coord=self.lon2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_1d_ignore_rowcoord(self):
        c_data, c_lons = add_cyclic_point(self.data2d, coord=self.lons,
                                          rowcoord=self.lat2d)
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_2d_rowcoord(self):
        c_data, c_lons, c_lats = add_cyclic_point(self.data2d,
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

    def test_data_and_coord_1d_already_cyclic_point(self):
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        c_data, c_lons = add_cyclic_point(r_data, coord=r_lons)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_2d_already_cyclic_point(self):
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        c_data, c_lons = add_cyclic_point(r_data, coord=r_lons)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_2d_rowcoord_already_cyclic_point(self):
        r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        r_lats = np.concatenate((self.lat2d, self.lat2d[:, -1:]), axis=1)
        c_data, c_lons, c_lats = add_cyclic_point(r_data, coord=r_lons,
                                                  rowcoord=r_lats)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_data_only_with_axis(self):
        c_data = add_cyclic_point(self.data4d, axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_only_ignore_rowcoord_with_axis(self):
        c_data = add_cyclic_point(self.data4d, axis=1, rowcoord=self.lat2d)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_data_and_coord_1d_with_axis(self):
        c_data, c_lons = add_cyclic_point(self.data4d, coord=self.lons, axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_2d_with_axis(self):
        c_data, c_lons = add_cyclic_point(self.data4d, coord=self.lon2d,
                                          axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_1d_ignore_rowcoord_with_axis(self):
        c_data, c_lons = add_cyclic_point(self.data4d, coord=self.lons,
                                          rowcoord=self.lat2d, axis=1)
        r_data = np.concatenate((self.data4d, self.data4d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lons, np.array([360.])))
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)

    def test_data_and_coord_2d_rowcoord_with_axis(self):
        c_data, c_lons, c_lats = add_cyclic_point(self.data4d,
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

    def test_masked_data(self):
        new_data = ma.masked_less(self.data2d, 3)
        c_data = add_cyclic_point(new_data)
        r_data = ma.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        assert_array_equal(c_data, r_data)

    def test_masked_data_and_coord_2d_rowcoord(self):
        new_data = ma.masked_less(self.data2d, 3)
        c_data, c_lons, c_lats = add_cyclic_point(new_data,
                                                  coord=self.lon2d,
                                                  rowcoord=self.lat2d)
        r_data = ma.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
        r_lons = np.concatenate((self.lon2d,
                                 np.full((self.lon2d.shape[0], 1), 360.)),
                                axis=1)
        r_lats = np.concatenate((self.lat2d, self.lat2d[:, -1:]), axis=1)
        assert_array_equal(c_data, r_data)
        assert_array_equal(c_lons, r_lons)
        assert_array_equal(c_lats, r_lats)

    def test_invalid_coord_dimensionality(self):
        lons3d = np.repeat(self.lon2d[np.newaxis], 3, axis=0)
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic_point(self.data2d, coord=lons3d)

    def test_invalid_rowcoord_dimensionality(self):
        with pytest.raises(ValueError):
            c_data, c_lons, c_lats = add_cyclic_point(self.data2d,
                                                      coord=self.lon2d,
                                                      rowcoord=self.lats)

    def test_invalid_coord_size_1d(self):
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic_point(self.data2d,
                                              coord=self.lons[:-1])

    def test_invalid_coord_size_2d(self):
        with pytest.raises(ValueError):
            c_data, c_lons = add_cyclic_point(self.data2d,
                                              coord=self.lon2d[:, :-1])

    def test_invalid_rowcoord_size(self):
        with pytest.raises(ValueError):
            c_data, c_lons, c_lats = add_cyclic_point(
                self.data2d, coord=self.lon2d, rowcoord=self.lat2d[:, 1:])

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            add_cyclic_point(self.data2d, axis=-3)
