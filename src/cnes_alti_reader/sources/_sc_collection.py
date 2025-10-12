"""Swot calval collection source."""

from __future__ import annotations

import dataclasses as dc
import datetime
from typing import TYPE_CHECKING

import fsspec
import numpy as np
import pandas as pd
import swot_calval.io as sc_io
import xarray as xr

from cnes_alti_reader.utilities import (
    dataset_select_field_1d,
    normalize_file_system,
    normalize_polygon,
)

from ._model import DOC_PARAMETERS_ALTI_SOURCE, CnesAltiSource, CnesAltiVariable

if TYPE_CHECKING:
    import geopandas as gpd_t
    import pyinterp.geodetic as pyi_geo_t
    import shapely.geometry as shg_t


@dc.dataclass(kw_only=True)
class ScCollectionSource(CnesAltiSource):
    __doc__ = f"""Source implementation for Swot Calval collections.

    Parameters
    ----------
    path
        Collection's path.

    {DOC_PARAMETERS_ALTI_SOURCE}
    """

    path: str
    fs: fsspec.AbstractFileSystem | str | None = dc.field(default=None, compare=False)

    time: str = "time"
    longitude: str = "longitude"
    latitude: str = "latitude"
    index: str = "num_lines"

    _collection: sc_io.Collection = dc.field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        self.path = str(self.path)
        self.fs = normalize_file_system(fs=self.fs)
        self._collection = sc_io.Collection(folder=self.path, filesystem=self.fs)

    def variables(self) -> dict[str, CnesAltiVariable]:
        if self._fields is not None:
            return self._fields

        self._fields = {}

        for f in self._collection.variables().values():
            attrs = {attr.name: attr.value for attr in f.attrs}
            self._fields[f.name] = CnesAltiVariable(
                name=f.name,
                units=attrs.get("units", ""),
                description=attrs.get("comment", ""),
            )

        return self._fields

    def period(self) -> tuple[np.datetime64, np.datetime64]:
        return self._collection.period()

    def half_orbit_periods(
        self,
        ho_min: tuple[int, int] | None = None,
        ho_max: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        data = self._collection.half_orbit_periods()
        mask = np.full(len(data), True)

        if ho_min is not None:
            mask = (
                (data["cycle_number"] == ho_min[0]) & (data["pass_number"] >= ho_min[1])
            ) | (data["cycle_number"] > ho_min[0])

        if ho_max is not None:
            mask &= (
                (data["cycle_number"] == ho_max[0]) & (data["pass_number"] <= ho_max[1])
            ) | (data["cycle_number"] < ho_max[0])

        return pd.DataFrame(data[mask])

    def query_date(
        self,
        start: np.datetime64,
        end: np.datetime64,
        variables: list[str] | None = None,
        polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None = None,
    ) -> xr.Dataset:
        polygon_gpd, _ = self._polygons(polygon=polygon)

        data = self._collection.query(
            first_day=start.astype("datetime64[D]").astype(datetime.date),
            last_day=end.astype("datetime64[D]").astype(datetime.date),
            selected_variables=variables,
            polygon=None,
        )

        if data is None:
            return self._empty_dataset()

        data = dataset_select_field_1d(
            data=data.to_xarray(),
            field=self.time,
            values=(start, end),
            include_end=True,
        )
        return self.restrict_to_polygon(data=data, polygon=polygon_gpd)

    def query_orbit(
        self,
        cycles_nb: int | list[int],
        passes_nb: int | list[int] | None = None,
        variables: list[str] | None = None,
        polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None = None,
    ) -> xr.Dataset:
        polygon_gpd, _ = self._polygons(polygon=polygon)

        data = self._collection.query(
            cycle_numbers=cycles_nb,
            pass_numbers=passes_nb,
            selected_variables=variables,
            polygon=None,
        )

        if data is None:
            return self._empty_dataset()

        return self.restrict_to_polygon(data=data.to_xarray(), polygon=polygon_gpd)

    @staticmethod
    def _polygons(
        polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None,
    ) -> tuple[gpd_t.GeoDataFrame | None, pyi_geo_t.Polygon | None]:
        """Normalize provided polygon to a GeoDataFrame and, if possible, a
        pyinterp.geodetic.Polygon."""
        if polygon is None:
            return None, None

        polygon_gpd = normalize_polygon(polygon=polygon)

        try:
            import pyinterp.geodetic as pyi_geo
        except ImportError:  # pragma: no cover
            return polygon_gpd, None

        return polygon_gpd, pyi_geo.Polygon.read_wkt(polygon_gpd.loc[0, "geometry"].wkt)
