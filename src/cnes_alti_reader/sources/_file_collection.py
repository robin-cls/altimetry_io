from __future__ import annotations

import dataclasses as dc
import enum
from typing import TYPE_CHECKING, Any

import fsspec
import numpy as np
import ocean_tools.io as ot_io
import ocean_tools.missions as ot_mis
import ocean_tools.swath.io as ot_sw_io
import ocean_tools.time as ot_time
import pandas as pd
import xarray as xr

from ..utilities import normalize_enum, normalize_file_system, polygon_bounding_box
from ._model import (
    CONST_CYCLE_NUMBER,
    CONST_END_TIME,
    CONST_PASS_NUMBER,
    CONST_START_TIME,
    DOC_PARAMETERS_ALTI_SOURCE,
    HALF_ORBIT_DTYPE,
    CnesAltiSource,
    CnesAltiVariable,
)

if TYPE_CHECKING:
    import geopandas as gpd_t
    import shapely.geometry as shg_t

OT_CONST_TIME = "time"
OT_CONST_CYCLE_NUMBER = "cycle_number"
OT_CONST_PASS_NUMBER = "pass_number"


class FCollectionType(enum.Enum):
    """Different types of file's collections."""

    SWOT_L2_LR_SSH = enum.auto()
    SWOT_L3_LR_SSH = enum.auto()
    SWOT_L3_LR_WIND_WAVE = enum.auto()
    NADIR_L2 = enum.auto()
    NADIR_L3 = enum.auto()

    def ot_database(self) -> type[ot_io.FilesDatabase]:
        known_collections = {
            FCollectionType.SWOT_L2_LR_SSH: ot_sw_io.NetcdfFilesDatabaseSwotLRL2,
            FCollectionType.SWOT_L3_LR_SSH: ot_sw_io.NetcdfFilesDatabaseSwotLRL3,
            FCollectionType.SWOT_L3_LR_WIND_WAVE: ot_sw_io.NetcdfFilesDatabaseSwotLRWW,
            FCollectionType.NADIR_L2: ot_sw_io.NetcdfFilesDatabaseL2Nadir,
            FCollectionType.NADIR_L3: ot_sw_io.NetcdfFilesDatabaseL3Nadir,
        }

        return known_collections[self]


@dc.dataclass(kw_only=True)
class FileCollectionSource(CnesAltiSource):
    __doc__ = f"""Source implementation for sets of files.

    Parameters
    ----------
    path
        File collection's path.

    {DOC_PARAMETERS_ALTI_SOURCE}
    """

    path: str
    fs: fsspec.AbstractFileSystem | str | None = dc.field(default=None, compare=False)
    ftype: FCollectionType

    subset: ot_sw_io.ProductSubset | None = None
    version: str | None = None

    # Just Nadir
    mission: ot_mis.MissionsPhases | None = None

    _database: ot_io.FilesDatabase = dc.field(repr=False, init=False, compare=False)
    _initialized: bool = dc.field(repr=False, init=False, compare=False)

    _with_ho: bool = dc.field(repr=False, init=False, compare=False, default=False)
    _data: pd.DataFrame = dc.field(repr=False, init=False, compare=False)

    def __post_init__(self):
        self.fs = normalize_file_system(fs=self.fs)
        self.ftype = normalize_enum(self.ftype, FCollectionType)
        self.subset = normalize_enum(self.subset, ot_sw_io.ProductSubset)

        self._database = self.ftype.ot_database()(path=self.path, fs=self.fs)

    def _request_kwargs(self) -> dict[str, Any]:
        request_kw = {}

        if self.subset is not None:
            request_kw["subset"] = self.subset

        if self.version is not None:
            request_kw["version"] = self.version

        if self.mission is not None:
            request_kw["mission"] = self.mission

        return request_kw

    def _initialize(self):
        if self._initialized:
            return

        pd_files: pd.DataFrame = self._database.list_files(
            sort=True, **self._request_kwargs()
        )

        periods = np.array([[p.start, p.stop] for p in pd_files[OT_CONST_TIME].values])

        if OT_CONST_CYCLE_NUMBER in pd_files.columns:
            self._with_ho = True

            ho_data = pd_files[[OT_CONST_PASS_NUMBER, OT_CONST_PASS_NUMBER]].values

            data = np.empty(ho_data.shape[0], dtype=HALF_ORBIT_DTYPE)
            data[CONST_CYCLE_NUMBER] = ho_data[:, 0]
            data[CONST_PASS_NUMBER] = ho_data[:, 1]
            data[CONST_START_TIME] = periods[:, 0]
            data[CONST_END_TIME] = periods[:, 1]

            self._data = pd.DataFrame(data)
        else:
            self._with_ho = False

            data = np.empty(periods.shape[0], dtype=HALF_ORBIT_DTYPE[-2:])
            data[CONST_START_TIME] = periods[:, 0]
            data[CONST_END_TIME] = periods[:, 1]

            self._data = pd.DataFrame(data)

        self._initialized = True

    def variables(self) -> dict[str, CnesAltiVariable]:
        if self._fields is not None:
            return self._fields

        self._fields = {}

        info: ot_io.GroupMetadata = self._database.variables_info()

        if info.subgroups:
            msg = "This collection contains subgroups. This is not supported yet."
            raise ValueError(msg)

        for variable in info.variables:
            self._fields[variable.name] = CnesAltiVariable(
                name=variable.name,
                units=variable.attributes.get("units", ""),
                description=variable.attributes.get("comment", ""),
            )

        return self._fields

    def period(self) -> tuple[np.datetime64, np.datetime64]:
        self._initialize()

        return np.min(self._data[CONST_START_TIME]), np.max(self._data[CONST_END_TIME])

    def half_orbit_periods(
        self,
        ho_min: tuple[int, int] | None = None,
        ho_max: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        self._check_orf()

        return self._data

    def query_date(
        self,
        start: np.datetime64,
        end: np.datetime64,
        variables: list[str] | None = None,
        polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None = None,
    ) -> xr.Dataset:
        bbox = polygon_bounding_box(polygon=polygon)

        data = self._database.query(
            time=ot_time.Period(start=start, stop=end),
            bbox=bbox,
            selected_variables=variables,
            **self._request_kwargs(),
        )

        data = data or self._empty_dataset()

        return self.restrict_to_polygon(data=data, polygon=polygon)

    def query_orbit(
        self,
        cycles_nb: int | list[int],
        passes_nb: int | list[int] | None = None,
        variables: list[str] | None = None,
        polygon: str | gpd_t.GeoDataFrame | shg_t.Polygon | None = None,
    ) -> xr.Dataset:
        self._check_orf()

        bbox = polygon_bounding_box(polygon=polygon)

        data = self._database.query(
            cycle_number=cycles_nb,
            pass_number=passes_nb,
            bbox=bbox,
            selected_variables=variables,
            **self._request_kwargs(),
        )

        data = data or self._empty_dataset()

        return self.restrict_to_polygon(data=data, polygon=polygon)

    def _check_orf(self):
        self._initialize()

        if not self._with_ho:
            msg = "This collection does not contain half-orbit information."
            raise ValueError(msg)
