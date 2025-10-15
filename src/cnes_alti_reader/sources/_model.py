from __future__ import annotations

import abc
import dataclasses as dc
import logging
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import pandas as pd
import xarray as xr

from cnes_alti_reader.utilities import restrict_to_polygon

if TYPE_CHECKING:
    import geopandas as gpd_t
    import shapely.geometry as shg_t

LOGGER = logging.getLogger(__name__)

CONST_CYCLE_NUMBER = "cycle_number"
CONST_PASS_NUMBER = "pass_number"
CONST_START_TIME = "start_time"
CONST_END_TIME = "end_time"

HALF_ORBIT_DTYPE = np.dtype(
    [
        (CONST_CYCLE_NUMBER, np.uint16),
        (CONST_PASS_NUMBER, np.uint16),
        (CONST_START_TIME, "M8[ns]"),
        (CONST_END_TIME, "M8[ns]"),
    ]
)


@dc.dataclass(kw_only=True)
class CnesAltiVariable:
    """A variable with its units and description."""

    name: str
    units: str = ""
    description: str = ""


T = TypeVar("T")

DOC_PARAMETERS_ALTI_SOURCE = """
    time
        Name of the time variable.
    longitude
        Name of the longitude variable.
    latitude
        Name of the latitude variable.
    index
        Name of the index dimension.
""".strip()


@dc.dataclass(kw_only=True)
class CnesAltiSource(Generic[T], abc.ABC):
    __doc__ = f"""Altimetric data source interface.

    Parameters
    ----------
    {DOC_PARAMETERS_ALTI_SOURCE}
    """
    time: str
    longitude: str
    latitude: str
    index: str

    _fields: dict[str, CnesAltiVariable] | None = dc.field(
        default=None, init=False, repr=False
    )

    @property
    @abc.abstractmethod
    def handler(self) -> T:
        """Source's handler."""

    @abc.abstractmethod
    def variables(self) -> dict[str, CnesAltiVariable]:
        """Variables contained in this source."""

    @abc.abstractmethod
    def period(self) -> tuple[np.datetime64, np.datetime64]:
        """Period covered by this source."""

    @abc.abstractmethod
    def half_orbit_periods(
        self,
        ho_min: tuple[int, int] | None = None,
        ho_max: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Half orbit periods covered by this altimetric data source.

        Parameters
        ----------
        ho_min
            Tuple of (cycle_number, pass_number) for the minimum half orbit.
        ho_max
            Tuple of (cycle_number, pass_number) for the maximum half orbit.

        Returns
        -------
        :
            Set of half orbit periods.
        """

    @abc.abstractmethod
    def query_date(
        self,
        start: np.datetime64,
        end: np.datetime64,
        variables: list[str] | None = None,
        polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None = None,
    ) -> xr.Dataset:
        """Query data between two dates.

        Parameters
        ----------
        start
            Starting date.
        end
            Ending date.
        variables
            Set of variables to query.
        polygon
            Selection polygon on which to reduce the data.

        Returns
        -------
        :
            Dataset respecting the query constraints.
        """

    def query_periods(
        self,
        periods: list[tuple[np.datetime64, np.datetime64]],
        variables: list[str] | None = None,
        polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None = None,
    ) -> xr.Dataset:
        """Query data contained in a set of periods.

        Parameters
        ----------
        periods
            Periods to query.
        variables
            Set of variables to query.
        polygon
            Selection polygon on which to reduce the data.

        Returns
        -------
        :
            Dataset respecting the query constraints.
        """
        data = [
            self.query_date(start=p[0], end=p[1], variables=variables, polygon=polygon)
            for p in periods
        ]

        return xr.concat(data, dim=self.index)

    @abc.abstractmethod
    def query_orbit(
        self,
        cycles_nb: int | list[int],
        passes_nb: int | list[int] | None = None,
        variables: list[str] | None = None,
        polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None = None,
    ) -> xr.Dataset:
        """Query data for a set of cycles and passes.

        Parameters
        ----------
        cycles_nb
            Cycle number or list of cycle numbers.
        passes_nb
            Passes number or list of passes numbers.
        variables
            Set of variables to query.
        polygon
            Selection polygon on which to reduce the data.

        Returns
        -------
        :
            Dataset respecting the query constraints.
        """

    def restrict_to_polygon(
        self, data: xr.Dataset, polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None
    ) -> xr.Dataset:
        """Apply a polygon selection to the provided data.

        Parameters
        ----------
        data
            Data from which to select data.
        polygon
            Selection polygon on which to reduce the data.

        Returns
        -------
        :
            Dataset reduced to the polygon constraints.
        """
        if polygon is None:
            return data

        return restrict_to_polygon(
            data=data,
            polygon=polygon,
            index=self.index,
            longitude=self.longitude,
            latitude=self.latitude,
        )

    def _empty_dataset(self) -> xr.Dataset:
        """Return an empty dataset with the correct index."""
        return xr.Dataset({self.index: np.array([], dtype="datetime64[ns]")})
