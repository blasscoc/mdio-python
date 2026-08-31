"""Side-scan sonar 2D waterfall MDIO dataset templates."""

from typing import Any

from mdio.builder.templates.base import AbstractDatasetTemplate
from mdio.builder.templates.types import SeismicDataDomain


class SideScan2DTemplate(AbstractDatasetTemplate):
    """Side-scan sonar 2D waterfall (time = across-track sample index).

    Dimensions are ``ping`` (along-track) × ``time`` (across-track sample).
    Physical coordinates reuse ``cdp_x`` / ``cdp_y`` for sensor easting/northing
    (or lon/lat when the source file's nav units are geographic) so catalog
    footprint helpers that expect those names keep working.
    """

    def __init__(self, data_domain: SeismicDataDomain = "time"):
        super().__init__(data_domain=data_domain)

        self._dim_names = ("ping", self._data_domain)
        self._physical_coord_names = ("cdp_x", "cdp_y")
        self._var_chunk_shape = (256, 1024)

    @property
    def _name(self) -> str:
        return f"SideScan2D{self._data_domain.capitalize()}"

    def _load_dataset_attributes(self) -> dict[str, Any]:
        return {
            "surveyType": "side_scan_sonar",
            "gatherType": "sidescan",
            "sourceFormat": "XTF",
        }
