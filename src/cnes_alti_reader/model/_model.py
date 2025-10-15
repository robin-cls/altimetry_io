from __future__ import annotations

import dataclasses as dc
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from cnes_alti_reader.sources import CnesAltiSource, CnesAltiVariable

if TYPE_CHECKING:
    import geopandas as gpd_t
    import shapely.geometry as shg_t


LOGGER = logging.getLogger(__name__)


@dc.dataclass
class CnesAltiData:
    source: CnesAltiSource

    @property
    def handler(self) -> Any:
        return self.source.handler

    def variables(self) -> dict[str, CnesAltiVariable]:
        """Variables contained in this altimetric data source."""
        return self.source.variables()

    def show_variables(self, containing: str | None = None) -> pd.DataFrame:
        """Display variables containing a given string as a DataFrame.

        Parameters
        ----------
        containing
            String contained in variable names or descriptions.

        Returns
        -------
        :
            List of variables as a DataFrame.
        """
        if containing is not None:
            containing = containing.upper()
            variables = {
                v.name: v
                for v in self.variables().values()
                if (containing in v.name.upper())
                or (containing in v.description.upper())
            }
        else:
            variables = self.variables()

        if not variables:
            data = pd.DataFrame([], columns=["name", "description", "units"])
        else:
            data = pd.DataFrame(
                np.array(
                    [[v.name, v.description, v.units] for v in variables.values()]
                ),
                columns=["name", "description", "units"],
            )

        return data

    def period(self) -> tuple[np.datetime64, np.datetime64]:
        """Period covered by this altimetric data source."""
        return self.source.period()

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
        return self.source.half_orbit_periods(ho_min=ho_min, ho_max=ho_max)

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
        return self.source.query_date(
            start=start, end=end, variables=variables, polygon=polygon
        )

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
        return self.source.query_periods(
            periods=periods, variables=variables, polygon=polygon
        )

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
        return self.source.query_orbit(
            cycles_nb=cycles_nb,
            passes_nb=passes_nb,
            variables=variables,
            polygon=polygon,
        )
