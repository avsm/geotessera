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

from .registry import zarr_store_url

log = logging.getLogger(__name__)

DEFAULT_STORE = zarr_store_url("v1")

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


def _zone_for_point(x: float, y: float, crs: str = "EPSG:4326") -> int:
    """Determine UTM zone for a point in any CRS.

    For UTM EPSG codes (326xx/327xx), extracts the zone directly.
    Otherwise projects to WGS84 and computes from longitude.
    """
    from pyproj import CRS as ProjCRS

    if crs != "EPSG:4326":
        proj_crs = ProjCRS.from_user_input(crs)
        utm_zone = proj_crs.utm_zone
        if utm_zone is not None:
            return int(utm_zone[:-1])
        lon, _ = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    else:
        lon = x
    return _zone_for_lon(lon)


# Tiles are 0.1 degrees and UTM zones 6 degrees, so round coordinates land on
# tile edges and multiples of 6 also land on zone seams. Reprojection rounds
# away from a shared edge, leaving occasional one-pixel unwritten gaps, and a
# zone's tiles stop at its boundary though its grid extends past it.

SEAM_SEARCH_PX = 1  # nearest-valid-pixel radius
SEAM_DEGREES = 1.0  # proximity to a seam before neighbouring zones are tried

VALID, WATER, NODATA, OUTSIDE = "valid", "water", "nodata", "outside"


def _seam_neighbours(lon: float) -> List[int]:
    """Zones to try after the one containing *lon*.  Empty away from a seam."""
    z = _zone_for_lon(lon)
    frac = (lon + 180.0) % 6.0
    out: List[int] = []
    if frac <= SEAM_DEGREES:
        out.append(60 if z == 1 else z - 1)
    if frac >= 6.0 - SEAM_DEGREES:
        out.append(1 if z == 60 else z + 1)
    return out


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
        case (int(), None, None):
            z = zone
        case (None, float(), None):
            z = _zone_for_lon(lon)
        case (None, None, tuple()):
            z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
        case _:
            raise TypeError("Provide exactly one of zone=, lon=, or bbox=")

    log.debug("open_zone: utm%02d from %s", z, store_url)
    ds = xr.open_zarr(
        store_url,
        group=f"utm{z:02d}",
        zarr_format=3,
        consolidated=True,
        chunks=SHARD_CHUNKS,
        **kwargs,
    )

    return ds


# ---------------------------------------------------------------------------
# xarray accessor
# ---------------------------------------------------------------------------


@xr.register_dataset_accessor("tessera")
class TesseraAccessor:
    """Tessera-aware methods on an xarray Dataset from a zarr zone.

    Uses coordinate-based selection (``sel(method='nearest')``) for all
    spatial lookups — no manual affine math.  Reads ``proj:code`` and
    ``spatial:transform`` from Dataset attrs, years from the time coordinate.
    """

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
        attrs = ds.attrs
        self._epsg: int = int(attrs["proj:code"].split(":")[1])
        self._to_utm = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{self._epsg}",
            always_xy=True,
        )
        # Derive years from the time coordinate
        if "time" in ds.coords:
            self._years: list[int] = [int(v) for v in ds.coords["time"].values]
        else:
            self._years = []
        # Read n_bands from geoemb:dimensions if available, else from band dim
        self._n_bands: int = int(
            attrs.get("geoemb:dimensions", ds.sizes.get("band", 128))
        )
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
        return self._n_bands

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

    def sample_at(
        self,
        x: float,
        y: float,
        year: int,
        *,
        crs: str = "EPSG:4326",
        search_px: int = SEAM_SEARCH_PX,
    ) -> np.ndarray:
        """Sample a single dequantised embedding.  Returns ``(B,)`` float32.

        Args:
            x: Easting or longitude.
            y: Northing or latitude.
            year: Year to sample.
            crs: Input coordinate CRS (default WGS84).  Accepts any
                pyproj-compatible CRS string.
            search_px: Nearest-valid-pixel radius for unwritten pixels;
                0 disables.  See :meth:`probe`.
        """
        if crs == f"EPSG:{self._epsg}":
            e, n = x, y
        elif crs == "EPSG:4326":
            e, n = self._to_utm.transform(x, y)
        else:
            proj = Transformer.from_crs(crs, f"EPSG:{self._epsg}", always_xy=True)
            e, n = proj.transform(x, y)

        log.debug("sample_at(%.6f, %.6f, crs=%s) → UTM(%.1f, %.1f)", x, y, crs, e, n)

        vec, _status = self.probe(e, n, year, search_px=search_px)
        if vec is None:
            return np.full(self.n_bands, np.nan, dtype=np.float32)
        return vec

    def probe(
        self,
        e: float,
        n: float,
        year: int,
        *,
        search_px: int = SEAM_SEARCH_PX,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Sample at UTM ``(e, n)``, reporting why when there is no value.

        Returns ``(embedding, status)``, status one of ``valid``, ``water``,
        ``nodata`` (never written) or ``outside`` (beyond this zone's grid).

        ``search_px`` is the radius within which an unwritten pixel falls back
        to the nearest valid one, covering the one-pixel seams left at tile
        corners; 0 disables it.  Water returns ``water`` rather than being
        searched past, so the fallback cannot report land for a sea location.
        """
        xname, yname = ("xc", "yc") if "xc" in self._ds.coords else ("x", "y")
        xs = self._ds.coords[xname].values
        ys = self._ds.coords[yname].values
        xi = int(np.abs(xs - e).argmin())
        yi = int(np.abs(ys - n).argmin())

        # sel(method="nearest") would snap a distant point to an edge pixel.
        if abs(xs[xi] - e) > self._px or abs(ys[yi] - n) > self._px:
            return None, OUTSIDE

        r = max(0, int(search_px))
        x0, x1 = max(0, xi - r), min(len(xs), xi + r + 1)
        y0, y1 = max(0, yi - r), min(len(ys), yi + r + 1)
        win = self._ds.isel({xname: slice(x0, x1), yname: slice(y0, y1)}).sel(time=year)
        scales = np.asarray(win["scales"].values, dtype=np.float64)
        ci, cj = yi - y0, xi - x0

        centre = scales[ci, cj]
        if np.isnan(centre):
            return None, WATER
        if np.isfinite(centre):
            bi, bj = ci, cj
        else:
            rows, cols = np.nonzero(np.isfinite(scales))
            if not len(rows):
                return None, NODATA
            k = int(np.argmin((rows - ci) ** 2 + (cols - cj) ** 2))
            bi, bj = int(rows[k]), int(cols[k])
        emb = np.asarray(win["embeddings"].values)
        return emb[:, bi, bj].astype(np.float32) * float(scales[bi, bj]), VALID

    def sample_points(
        self,
        coords: List[Tuple[float, float]],
        year: int,
        *,
        crs: str = "EPSG:4326",
        progress: bool = True,
    ) -> np.ndarray:
        """Sample embeddings at points.  Returns ``(N, B)`` float32.

        Args:
            coords: List of (x, y) tuples in the given CRS.
            crs: Input coordinate CRS (default WGS84).
        """
        it = coords
        if progress:
            it = track(coords, description="Sampling points...", transient=True)
        return np.array([self.sample_at(x, y, year, crs=crs) for x, y in it])

    # -- Region reading -----------------------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        crs: str = "EPSG:4326",
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Read and dequantise a bbox region.

        Args:
            bbox: (x_min, y_min, x_max, y_max) in the given CRS.
            crs: Input bbox CRS (default WGS84).

        Returns ``(mosaic, transform)`` where mosaic is ``(H, W, B)``
        float32 and transform is a rasterio Affine for the window.
        """
        zone_crs = f"EPSG:{self._epsg}"
        if crs == zone_crs:
            e_nw, n_nw = bbox[0], bbox[3]
            e_se, n_se = bbox[2], bbox[1]
        elif crs == "EPSG:4326":
            e_nw, n_nw = self._to_utm.transform(bbox[0], bbox[3])
            e_se, n_se = self._to_utm.transform(bbox[2], bbox[1])
        else:
            proj = Transformer.from_crs(crs, zone_crs, always_xy=True)
            e_nw, n_nw = proj.transform(bbox[0], bbox[3])
            e_se, n_se = proj.transform(bbox[2], bbox[1])
        e_min, e_max = min(e_nw, e_se), max(e_nw, e_se)
        n_min, n_max = min(n_nw, n_se), max(n_nw, n_se)

        # y is descending (north→south), so slice is (n_max, n_min)
        sub = self._ds.sel(time=year, x=slice(e_min, e_max), y=slice(n_max, n_min))
        h, w = int(sub.sizes["y"]), int(sub.sizes["x"])
        log.info(
            "read_region: %d x %d pixels (%s), %.0fm resolution",
            h,
            w,
            f"{h * w:,}",
            self._px,
        )

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
            TESSERA store at ``data.source.coop/tessera/tessera/zarr``.

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
        self.model_version: str = root_attrs.get("geoemb:model", "")
        self.build_version: str = root_attrs.get("geoemb:build_version", "")
        self.n_bands: int = int(root_attrs.get("geoemb:dimensions", 128))
        # Derive years from the first zone's time coordinate array
        self.years: list[int] = []
        for member_name in sorted(root.keys()):
            if member_name.startswith("utm"):
                try:
                    zone_grp = root[member_name]
                    time_arr = zone_grp["time"][:]
                    self.years = [int(v) for v in time_arr]
                    break
                except Exception:
                    continue
        self._cache: dict[int, xr.Dataset] = {}
        log.info(
            "GeoTesseraZarr: %s, years=%s, model=%s",
            self.url,
            self.years,
            self.model_version,
        )

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
            case (int(), None, None):
                z = zone
            case (None, float(), None):
                z = _zone_for_lon(lon)
            case (None, None, tuple()):
                z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
            case _:
                raise TypeError("Provide exactly one of zone=, lon=, or bbox=")

        if z not in self._cache:
            ds = open_zone(self.url, zone=z)
            self._cache[z] = ds
        return self._cache[z]

    # -- Point sampling (cross-zone) ----------------------------------------

    def sample_at(
        self,
        x: float,
        y: float,
        year: int,
        *,
        crs: str = "EPSG:4326",
        cross_zone: bool = True,
        search_px: int = SEAM_SEARCH_PX,
    ) -> np.ndarray:
        """Sample a single embedding, routing to the correct zone.

        Args:
            x: Easting or longitude.
            y: Northing or latitude.
            crs: Input CRS (default WGS84).
            cross_zone: Also try the neighbouring zone near a seam.  A tile
                belongs to the zone containing its centre, so a point on a
                seam is often covered by the zone next door.
            search_px: Nearest-valid-pixel radius; 0 disables.

        Returns ``(B,)`` float32, or a NaN row for open water and for
        locations outside coverage.  Use :meth:`probe` to tell those apart.
        """
        vec, status = self.probe(
            x, y, year, crs=crs, cross_zone=cross_zone, search_px=search_px
        )
        if vec is None:
            return np.full(self.n_bands, np.nan, dtype=np.float32)
        return vec

    def probe(
        self,
        x: float,
        y: float,
        year: int,
        *,
        crs: str = "EPSG:4326",
        cross_zone: bool = True,
        search_px: int = SEAM_SEARCH_PX,
    ) -> Tuple[Optional[np.ndarray], str]:
        """:meth:`sample_at`, reporting why when there is no value.

        Returns ``(embedding, status)``; see :meth:`TesseraAccessor.probe`.
        Use this instead of testing ``sample_at`` for NaN to tell open water
        from a location missing from the store.
        """
        lon = x
        if crs != "EPSG:4326":
            lon, _ = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
                x, y
            )

        zones = [_zone_for_point(x, y, crs)]
        if cross_zone:
            zones += [z for z in _seam_neighbours(lon) if z not in zones]

        seen = set()
        for z in zones:
            try:
                acc = self.open_zone(zone=z).tessera
            except Exception as exc:  # zone absent from this store
                log.debug("probe: zone %d unavailable (%s)", z, exc)
                continue
            if crs == acc.crs:
                e, n = x, y
            else:
                e, n = Transformer.from_crs(crs, acc.crs, always_xy=True).transform(
                    x, y
                )
            vec, status = acc.probe(e, n, year, search_px=search_px)
            if status == VALID:
                if z != zones[0]:
                    log.debug("probe: %.6f,%.6f served from zone %d", x, y, z)
                return vec, VALID
            seen.add(status)

        # Water is a real answer about the location; outside means no zone
        # grid covered it at all.
        for status in (WATER, NODATA):
            if status in seen:
                return None, status
        return None, OUTSIDE

    def sample_points(
        self,
        coords: List[Tuple[float, float]],
        year: int,
        *,
        crs: str = "EPSG:4326",
        progress: bool = True,
        cross_zone: bool = True,
        search_px: int = SEAM_SEARCH_PX,
    ) -> np.ndarray:
        """Sample embeddings at points, routing each to its zone.

        Args:
            coords: List of (x, y) tuples in the given CRS.
            crs: Input CRS (default WGS84).
            cross_zone: See :meth:`sample_at`.
            search_px: See :meth:`sample_at`.

        Returns ``(N, B)`` float32.  Points without an embedding get NaN rows.
        """
        it = coords
        if progress:
            it = track(coords, description="Sampling points...", transient=True)
        return np.array(
            [
                self.sample_at(
                    x, y, year, crs=crs, cross_zone=cross_zone, search_px=search_px
                )
                for x, y in it
            ]
        )

    # -- Region reading (dominant zone) -------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        crs: str = "EPSG:4326",
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """Read and dequantise a bbox region.

        Args:
            bbox: (x_min, y_min, x_max, y_max) in the given CRS.
            crs: Input bbox CRS (default WGS84).

        Uses the dominant UTM zone (from bbox centre).  Returns
        ``(mosaic, transform, crs)`` where mosaic is ``(H, W, B)`` float32.
        """
        # Convert bbox centre to WGS84 for zone routing
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        z = _zone_for_point(cx, cy, crs)

        ds = self.open_zone(zone=z)
        mosaic, transform = ds.tessera.read_region(
            bbox, year, crs=crs, progress=progress
        )
        return mosaic, transform, ds.tessera.crs
