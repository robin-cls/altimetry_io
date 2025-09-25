import abc
import dataclasses as dc

import numpy as np
import pyinterp.geodetic as pyi_geo
import xarray as xr


@dc.dataclass
class CnesAltiSource(abc.ABC):
    @abc.abstractmethod
    def list_variables(self) -> list[str]:
        pass

    @abc.abstractmethod
    def query_date(
        self,
        start: np.datetime64,
        end: np.datetime64,
        variables: list[str] | None = None,
        polygon: pyi_geo.Polygon | None = None,
    ) -> xr.Dataset:
        pass

    @abc.abstractmethod
    def query_periods(
        self,
        periods: list[tuple[np.datetime64, np.datetime64]],
        variables: list[str] | None = None,
        polygon: pyi_geo.Polygon | None = None,
    ) -> xr.Dataset:
        pass

    @abc.abstractmethod
    def query_cycle(
        self,
        cycles_nb: int | list[int],
        variables: list[str] | None = None,
        polygon: pyi_geo.Polygon | None = None,
    ) -> xr.Dataset:
        pass

    @abc.abstractmethod
    def query_cycle_pass(
        self,
        cycles_nb: int | list[int],
        passes_nb: int | list[int],
        variables: list[str] | None = None,
        polygon: pyi_geo.Polygon | None = None,
    ) -> xr.Dataset:
        pass


@dc.dataclass
class CnesAltiData:
    source: CnesAltiSource

    def list_variables(self) -> list[str]:
        return self.source.list_variables()

    def query_date(
        self,
        start: np.datetime64,
        end: np.datetime64,
        variables: list[str] | None = None,
        polygon: pyi_geo.Polygon | None = None,
    ) -> xr.Dataset:
        return self.source.query_date(
            start=start, end=end, variables=variables, polygon=polygon
        )

    def query_periods(
        self,
        periods: list[tuple[np.datetime64, np.datetime64]],
        variables: list[str] | None = None,
        polygon: pyi_geo.Polygon | None = None,
    ) -> xr.Dataset:
        return self.source.query_periods(
            periods=periods, variables=variables, polygon=polygon
        )

    def query_cycle(
        self,
        cycles_nb: int | list[int],
        variables: list[str] | None = None,
        polygon: pyi_geo.Polygon | None = None,
    ) -> xr.Dataset:
        return self.source.query_cycle(
            cycles_nb=cycles_nb, variables=variables, polygon=polygon
        )

    def query_cycle_pass(
        self,
        cycles_nb: int | list[int],
        passes_nb: int | list[int],
        variables: list[str] | None = None,
        polygon: pyi_geo.Polygon | None = None,
    ) -> xr.Dataset:
        return self.source.query_cycle_pass(
            cycles_nb=cycles_nb,
            passes_nb=passes_nb,
            variables=variables,
            polygon=polygon,
        )
