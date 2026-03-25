"""
GeoTesseraZarr — read embeddings from a Tessera zarr store.

Provides ``GeoTesseraZarr`` for store-level access (zone routing, point
sampling, region reading) and a ``tessera`` xarray accessor for per-zone
operations.  All spatial indexing uses xarray coordinate-based selection
(``sel(method='nearest')``), with no manual affine math.

Usage::

    from geotessera.store import GeoTesseraZarr

    gt = GeoTesseraZarr()  # default public store
    X = gt.sample_points([(-2.97, 53.44), (-2.96, 53.43)], year=2025)
    mosaic, transform, crs = gt.read_region(bbox, year=2025)

    # Direct zone access
    ds = gt.open_zone(lon=-2.97)
    emb = ds.tessera.sample_at(-2.97, 53.44, year=2025)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import rasterio.transform
import xarray as xr
import zarr
from pyproj import Transformer
from rich.progress import track

log = logging.getLogger(__name__)

DEFAULT_STORE = "https://dl2.geotessera.org/zarr/v2/store.zarr"

# Shard-aligned chunk sizes so dask tasks match zarr shards
SHARD_CHUNKS = {"time": 1, "band": 128, "y": 4096, "x": 4096}


def enable_http_logging(level: int = logging.DEBUG) -> None:
    """Enable fsspec HTTP request logging for debugging.

    Call before opening a store to see every HTTP request::

        from geotessera.store import enable_http_logging
        enable_http_logging()
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fsspec.http").setLevel(level)
    log.setLevel(level)


def _zone_for_lon(lon: float) -> int:
    """UTM zone number (1-60) for a WGS84 longitude."""
    return max(1, min(60, int(math.floor((lon + 180) / 6)) + 1))


def open_zone(
    store_url: str = DEFAULT_STORE,
    *,
    zone: Optional[int] = None,
    lon: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    **kwargs,
) -> xr.Dataset:
    """Open a tessera zone as an xarray Dataset.

    Provide exactly one of ``zone``, ``lon``, or ``bbox`` to select the
    UTM zone.  Returns a Dataset with the ``.tessera`` accessor.

    Args:
        store_url: Zarr store URL or local path.
        zone: UTM zone number (1-60).
        lon: A longitude — zone is derived automatically.
        bbox: (min_lon, min_lat, max_lon, max_lat) — zone from centre.

    Example::

        from geotessera.store import open_zone
        ds = open_zone(lon=-2.97)
        ds = open_zone(bbox=(-3.0, 53.4, -2.9, 53.5))
        ds = open_zone(zone=30)
    """
    match (zone, lon, bbox):
        case (int(), None, None):    z = zone
        case (None, float(), None):  z = _zone_for_lon(lon)
        case (None, None, tuple()):  z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
        case _: raise TypeError("Provide exactly one of zone=, lon=, or bbox=")

    log.debug("open_zone: utm%02d from %s", z, store_url)
    ds = xr.open_zarr(
        store_url,
        group=f"utm{z:02d}",
        zarr_format=3,
        consolidated=True,
        chunks=SHARD_CHUNKS,
        **kwargs,
    )

    # Attach computed piecewise tile transform — reproduces the tile
    # generation geometry (0.1° WGS84 → UTM projection) to compute
    # exact per-pixel coordinates without stored metadata.
    from .tile_transform import TesseraTileTransform
    try:
        transform = TesseraTileTransform.from_zone_attrs(ds.attrs)
        idx = xr.indexes.CoordinateTransformIndex(transform)
        ds = ds.assign_coords(xr.Coordinates.from_xindex(idx))
        log.debug("Attached TesseraTileTransform for EPSG:%d", transform.epsg)
    except Exception as exc:
        log.debug("Could not attach tile transform: %s", exc)

    return ds


# ---------------------------------------------------------------------------
# xarray accessor
# ---------------------------------------------------------------------------

@xr.register_dataset_accessor("tessera")
class TesseraAccessor:
    """Tessera-aware methods on an xarray Dataset from a zarr zone.

    Uses coordinate-based selection (``sel(method='nearest')``) for all
    spatial lookups — no manual affine math.  Reads ``proj:code``,
    ``spatial:transform``, and ``tessera:years`` from Dataset attrs.
    """

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
        attrs = ds.attrs
        self._epsg: int = int(attrs["proj:code"].split(":")[1])
        self._to_utm = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{self._epsg}", always_xy=True,
        )
        self._to_wgs = Transformer.from_crs(
            f"EPSG:{self._epsg}", "EPSG:4326", always_xy=True,
        )
        self._years: list[int] = list(attrs.get("tessera:years", []))
        t = attrs["spatial:transform"]
        self._px: float = float(t[0])
        log.debug("TesseraAccessor: EPSG:%d, years=%s", self._epsg, self._years)

    # -- Properties ---------------------------------------------------------

    @property
    def crs(self) -> str:
        """CRS string, e.g. ``'EPSG:32630'``."""
        return f"EPSG:{self._epsg}"

    @property
    def pixel_size(self) -> float:
        """Pixel size in CRS units (metres for UTM)."""
        return self._px

    @property
    def years(self) -> list[int]:
        """Available years, matching the time dimension order."""
        return self._years

    @property
    def n_bands(self) -> int:
        """Number of embedding bands."""
        return int(self._ds.sizes["band"])

    # -- Dequantisation -----------------------------------------------------

    @staticmethod
    def dequantise(emb_int8: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantise int8 embeddings: ``(B,H,W)`` + ``(H,W)`` → ``(H,W,B)`` float32.

        Non-finite scales (NaN = water, +inf = no data) produce NaN rows.
        """
        valid = np.isfinite(scales)
        safe = np.where(valid, scales, 0.0)
        f32 = emb_int8.astype(np.float32) * safe[np.newaxis, :, :]
        f32[:, ~valid] = np.nan
        return f32.transpose(1, 2, 0)

    # -- Point sampling -----------------------------------------------------

    def sample_at(self, lon: float, lat: float, year: int) -> np.ndarray:
        """Sample a single dequantised embedding.  Returns ``(B,)`` float32."""
        e, n = self._to_utm.transform(lon, lat)
        log.debug("sample_at(%.6f, %.6f) → UTM(%.1f, %.1f)", lon, lat, e, n)

        # Use tile transform coords (xc/yc) if available, else regular (x/y)
        if "xc" in self._ds.coords:
            import xarray as xr
            pixel = self._ds.sel(
                time=year,
                xc=xr.DataArray(e), yc=xr.DataArray(n),
                method="nearest",
            )
        else:
            pixel = self._ds.sel(time=year, x=e, y=n, method="nearest")
        scale = float(pixel["scales"].values)
        if not np.isfinite(scale):
            return np.full(self.n_bands, np.nan, dtype=np.float32)
        return pixel["embeddings"].values.astype(np.float32) * scale

    def sample_points(
        self,
        coords: List[Tuple[float, float]],
        year: int,
        *,
        progress: bool = True,
    ) -> np.ndarray:
        """Sample embeddings at ``(lon, lat)`` points.  Returns ``(N, B)`` float32."""
        it = coords
        if progress:
            it = track(coords, description="Sampling points...", transient=True)
        return np.array([self.sample_at(lon, lat, year) for lon, lat in it])

    # -- Region reading -----------------------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Read and dequantise a bbox region.

        Returns ``(mosaic, transform)`` where mosaic is ``(H, W, B)``
        float32 and transform is a rasterio Affine for the window.
        """
        e_nw, n_nw = self._to_utm.transform(bbox[0], bbox[3])
        e_se, n_se = self._to_utm.transform(bbox[2], bbox[1])
        e_min, e_max = min(e_nw, e_se), max(e_nw, e_se)
        n_min, n_max = min(n_nw, n_se), max(n_nw, n_se)

        # y is descending (north→south), so slice is (n_max, n_min)
        sub = self._ds.sel(time=year, x=slice(e_min, e_max), y=slice(n_max, n_min))
        h, w = int(sub.sizes["y"]), int(sub.sizes["x"])
        log.info("read_region: %d x %d pixels (%s), %.0fm resolution", h, w, f"{h*w:,}", self._px)

        if progress:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                scales = sub["scales"].values
                emb_int8 = sub["embeddings"].values
        else:
            scales = sub["scales"].values
            emb_int8 = sub["embeddings"].values

        mosaic = self.dequantise(emb_int8, scales)

        # Build affine from the selected window's coordinate values
        x0 = float(sub["x"].values[0]) - 0.5 * self._px  # pixel centre → corner
        y0 = float(sub["y"].values[0]) + 0.5 * self._px
        transform = rasterio.transform.Affine(self._px, 0, x0, 0, -self._px, y0)
        return mosaic, transform


# ---------------------------------------------------------------------------
# GeoTesseraZarr — store-level API with zone routing
# ---------------------------------------------------------------------------

class GeoTesseraZarr:
    """Read embeddings from a Tessera zarr store.

    Routes geographic queries to the correct UTM zone automatically.
    For single-zone work, use :func:`open_zone` directly.

    Args:
        store_url: Zarr store URL or local path.  Defaults to the public
            TESSERA store at ``dl2.geotessera.org``.

    Example::

        from geotessera.store import GeoTesseraZarr

        gt = GeoTesseraZarr()
        print(gt.years)  # [2017, 2018, ..., 2025]

        # Sample embeddings at points
        X = gt.sample_points([(-2.97, 53.44)], year=2025)

        # Read a region
        mosaic, transform, crs = gt.read_region(
            (-3.0, 53.4, -2.9, 53.5), year=2025,
        )
    """

    def __init__(self, store_url: str = DEFAULT_STORE):
        self.url = store_url.rstrip("/")
        root = zarr.open_group(self.url, mode="r")
        root_attrs = dict(root.attrs)
        self.years: list[int] = root_attrs.get("tessera:years", [])
        self.model_version: str = root_attrs.get("tessera:model_version", "")
        self.build_version: str = root_attrs.get("tessera:build_version", "")
        self._cache: dict[int, xr.Dataset] = {}
        log.info("GeoTesseraZarr: %s, years=%s, model=%s",
                 self.url, self.years, self.model_version)

    def __repr__(self) -> str:
        return f"GeoTesseraZarr({self.url!r}, years={self.years})"

    # -- Zone access --------------------------------------------------------

    def open_zone(
        self,
        *,
        zone: Optional[int] = None,
        lon: Optional[float] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> xr.Dataset:
        """Open a zone Dataset with the ``.tessera`` accessor.

        Provide exactly one of ``zone``, ``lon``, or ``bbox``.
        Datasets are cached for the lifetime of this instance.
        """
        match (zone, lon, bbox):
            case (int(), None, None):    z = zone
            case (None, float(), None):  z = _zone_for_lon(lon)
            case (None, None, tuple()):  z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
            case _: raise TypeError("Provide exactly one of zone=, lon=, or bbox=")

        if z not in self._cache:
            ds = open_zone(self.url, zone=z)
            ds.attrs.setdefault("tessera:years", self.years)
            self._cache[z] = ds
        return self._cache[z]

    # -- Point sampling (cross-zone) ----------------------------------------

    def sample_at(self, lon: float, lat: float, year: int) -> np.ndarray:
        """Sample a single embedding, routing to the correct zone.

        Returns ``(B,)`` float32.
        """
        ds = self.open_zone(lon=lon)
        return ds.tessera.sample_at(lon, lat, year)

    def sample_points(
        self,
        coords: List[Tuple[float, float]],
        year: int,
        *,
        progress: bool = True,
    ) -> np.ndarray:
        """Sample embeddings at ``(lon, lat)`` points, routing each to its zone.

        Returns ``(N, B)`` float32.  Points outside coverage get NaN rows.
        """
        it = coords
        if progress:
            it = track(coords, description="Sampling points...", transient=True)
        return np.array([
            self.open_zone(lon=lon).tessera.sample_at(lon, lat, year)
            for lon, lat in it
        ])

    # -- Region reading (dominant zone) -------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """Read and dequantise a bbox region.

        Uses the dominant UTM zone (from bbox centre).  Returns
        ``(mosaic, transform, crs)`` where mosaic is ``(H, W, B)`` float32.
        """
        z_w = _zone_for_lon(bbox[0])
        z_e = _zone_for_lon(bbox[2])
        if z_w != z_e:
            log.warning("Bbox spans UTM zones %d-%d, using zone %d (centre)",
                        z_w, z_e, _zone_for_lon((bbox[0] + bbox[2]) / 2))

        ds = self.open_zone(bbox=bbox)
        mosaic, transform = ds.tessera.read_region(bbox, year, progress=progress)
        return mosaic, transform, ds.tessera.crs
