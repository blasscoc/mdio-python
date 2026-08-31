"""Unit tests for SideScan2DTemplate."""

import pytest
from tests.unit.v1.helpers import validate_variable

from mdio.builder.schemas.chunk_grid import RegularChunkGrid
from mdio.builder.schemas.dtype import ScalarType
from mdio.builder.schemas.dtype import StructuredType
from mdio.builder.schemas.v1.units import LengthUnitEnum
from mdio.builder.schemas.v1.units import LengthUnitModel
from mdio.builder.schemas.v1.units import TimeUnitEnum
from mdio.builder.schemas.v1.units import TimeUnitModel
from mdio.builder.templates.side_scan_2d import SideScan2DTemplate
from mdio.builder.templates.types import SeismicDataDomain

UNITS_METER = LengthUnitModel(length=LengthUnitEnum.METER)
UNITS_SECOND = TimeUnitModel(time=TimeUnitEnum.SECOND)


def test_configuration() -> None:
    """Test configuration of SideScan2DTemplate."""
    t = SideScan2DTemplate(data_domain="time")

    assert t._data_domain == "time"
    assert t._dim_names == ("ping", "time")
    assert t._physical_coord_names == ("cdp_x", "cdp_y")
    assert t.full_chunk_shape == (256, 1024)
    assert t.name == "SideScan2DTime"
    assert t.default_variable_name == "amplitude"
    assert t._load_dataset_attributes() == {
        "surveyType": "side_scan_sonar",
        "gatherType": "sidescan",
        "sourceFormat": "XTF",
    }


def test_build_dataset(structured_headers: StructuredType) -> None:
    """Test building a complete side-scan 2D dataset."""
    t = SideScan2DTemplate(data_domain="time")
    t.add_units({"cdp_x": UNITS_METER, "cdp_y": UNITS_METER})
    t.add_units({"time": UNITS_SECOND})

    dataset = t.build_dataset("SSS Line 001", sizes=(64, 512), header_dtype=structured_headers)

    assert dataset.metadata.name == "SSS Line 001"
    assert dataset.metadata.attributes["surveyType"] == "side_scan_sonar"
    assert dataset.metadata.attributes["gatherType"] == "sidescan"
    assert dataset.metadata.attributes["sourceFormat"] == "XTF"

    validate_variable(dataset, name="headers", dims=[("ping", 64)], coords=["cdp_x", "cdp_y"], dtype=structured_headers)
    validate_variable(
        dataset,
        name="trace_mask",
        dims=[("ping", 64)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.BOOL,
    )
    validate_variable(dataset, name="ping", dims=[("ping", 64)], coords=["ping"], dtype=ScalarType.INT32)
    domain = validate_variable(dataset, name="time", dims=[("time", 512)], coords=["time"], dtype=ScalarType.INT32)
    assert domain.metadata.units_v1 == UNITS_SECOND

    cdp_x = validate_variable(dataset, name="cdp_x", dims=[("ping", 64)], coords=["cdp_x"], dtype=ScalarType.FLOAT64)
    assert cdp_x.metadata.units_v1 == UNITS_METER
    cdp_y = validate_variable(dataset, name="cdp_y", dims=[("ping", 64)], coords=["cdp_y"], dtype=ScalarType.FLOAT64)
    assert cdp_y.metadata.units_v1 == UNITS_METER

    v = validate_variable(
        dataset,
        name="amplitude",
        dims=[("ping", 64), ("time", 512)],
        coords=["cdp_x", "cdp_y"],
        dtype=ScalarType.FLOAT32,
    )
    assert isinstance(v.metadata.chunk_grid, RegularChunkGrid)
    assert v.metadata.chunk_grid.configuration.chunk_shape == (256, 1024)


@pytest.mark.parametrize("data_domain", ["Time", "time"])
def test_domain_case_handling(data_domain: str) -> None:
    """Test that domain parameter handles different cases correctly."""
    template = SideScan2DTemplate(data_domain=data_domain)  # type: ignore[arg-type]
    assert template._data_domain == data_domain.lower()
    assert template.name == f"SideScan2D{data_domain.lower().capitalize()}"
