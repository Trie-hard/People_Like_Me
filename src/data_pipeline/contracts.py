"""Pandera schemas for harmonized NHANES tables (PROJECT_PLAN.md §3.4)."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class MecCohortSchema(pa.DataFrameModel):
    seqn: Series[int] = pa.Field(ge=1)
    cycle: Series[str]
    ridageyr: Series[int] = pa.Field(ge=20, le=80)
    riagendr: Series[int] = pa.Field(isin=[1, 2])
    ridreth3: Series[float] = pa.Field(nullable=True)
    indfmpir: Series[float] = pa.Field(nullable=True, ge=0)
    sdmvstra: Series[int] = pa.Field(ge=1)
    sdmvpsu: Series[int] = pa.Field(ge=1)
    bmxbmi: Series[float] = pa.Field(ge=12, le=80)
    sbp: Series[float] = pa.Field(ge=60, le=260)
    dbp: Series[float] = pa.Field(ge=30, le=150)
    map: Series[float] = pa.Field(ge=40, le=200)
    pp: Series[float] = pa.Field(ge=10, le=120)
    lbxgh: Series[float] = pa.Field(nullable=True, ge=3, le=20)
    lbxtc: Series[float] = pa.Field(nullable=True, ge=50, le=600)
    lbdhdd: Series[float] = pa.Field(nullable=True, ge=10, le=150)
    diq010: Series[float] = pa.Field(nullable=True, isin=[1, 2, 3])
    wtmec6yr: Series[float] = pa.Field(gt=0)
    on_glucose_med: Series[bool] = pa.Field()
    on_lipid_med: Series[bool] = pa.Field()

    class Config:
        strict = False
        coerce = True


class FastingCohortSchema(MecCohortSchema):
    lbxglu: Series[float] = pa.Field(ge=40, le=500)
    lbxtr: Series[float] = pa.Field(nullable=True, ge=20, le=2000)
    wtsaf6yr: Series[float] = pa.Field(gt=0)
    phafsthr: Series[float] = pa.Field(ge=8, lt=24)
