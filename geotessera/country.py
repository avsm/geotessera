"""Country lookup functionality using Natural Earth data."""

from typing import Tuple, Optional, List, Dict, Callable
import geopandas as gpd
import zipfile
import os
from pathlib import Path
import difflib


class CountryLookup:
    """Provides country name to geometry and bounding box lookup using Natural Earth data."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ):
        """Initialize with optional cache directory for Natural Earth data.

        Args:
            cache_dir: Optional cache directory path
            progress_callback: Optional callback(current, total, status) for progress updates
        """
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            # Use platform-appropriate cache directory
            if os.name == 'nt':
                base = Path(os.environ.get('LOCALAPPDATA', '~')).expanduser()
            else:
                base = Path(os.environ.get('XDG_CACHE_HOME', '~/.cache')).expanduser()
            self._cache_dir = base / 'geotessera'

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._countries_gdf: Optional[gpd.GeoDataFrame] = None
        self._name_lookup: Optional[Dict[str, str]] = None
        self._progress_callback = progress_callback

    def _get_countries_data_path(self) -> Path:
        """Download and extract Natural Earth countries data, return path to GeoJSON."""
        # Check if data already exists
        extract_dir = self._cache_dir / "natural-earth-data"
        geojson_path = (
            extract_dir
            / "natural-earth-vector-5.1.2"
            / "geojson"
            / "ne_110m_admin_0_countries.geojson"
        )

        if geojson_path.exists():
            if self._progress_callback:
                self._progress_callback(100, 100, "Country data already cached")
            return geojson_path

        # Report download starting
        if self._progress_callback:
            self._progress_callback(0, 100, "Downloading country boundaries...")

        # Import the Rich downloader from progress module
        from .progress import RichHTTPDownloader

        # Create a wrapper around RichHTTPDownloader that adapts the progress reporting
        class CountryDataDownloader(RichHTTPDownloader):
            def __init__(self, progress_callback):
                # Don't call parent init, we'll handle progress ourselves
                self.outer_progress_callback = progress_callback

            def __call__(self, url, output_file, pooch):
                """Download with progress reporting."""
                import requests
                from pathlib import Path

                # Ensure output directory exists
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Report starting download
                if self.outer_progress_callback:
                    self.outer_progress_callback(0, 100, "Connecting to GitHub...")

                # Start download with streaming, follow redirects
                # Use a session for better connection handling
                session = requests.Session()
                session.headers.update(
                    {
                        "User-Agent": "GeoTessera/1.0 (https://github.com/tessera/geotessera)"
                    }
                )

                try:
                    response = session.get(
                        url, stream=True, allow_redirects=True, timeout=60
                    )
                    response.raise_for_status()

                    # Get total file size from headers
                    total_size = int(response.headers.get("content-length", 0))

                    # Initialize progress
                    downloaded = 0
                    last_mb = 0

                    # Download in chunks
                    chunk_size = 8192
                    with open(output_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:  # Filter out keep-alive chunks
                                f.write(chunk)
                                downloaded += len(chunk)

                                # Update progress only when we've downloaded another MB
                                mb_downloaded = downloaded / (1024 * 1024)
                                if int(mb_downloaded) > last_mb:
                                    last_mb = int(mb_downloaded)
                                    if self.outer_progress_callback:
                                        if total_size > 0:
                                            progress = int(
                                                (downloaded / total_size) * 50
                                            )  # First 50% for download
                                            mb_total = total_size / (1024 * 1024)
                                            self.outer_progress_callback(
                                                progress,
                                                100,
                                                f"Downloading country data... {mb_downloaded:.1f}/{mb_total:.1f} MB",
                                            )
                                        else:
                                            # No total size, estimate progress based on typical size (~18MB)
                                            estimated_total = (
                                                18 * 1024 * 1024
                                            )  # 18MB typical size
                                            progress = min(
                                                45,
                                                int(
                                                    (downloaded / estimated_total) * 45
                                                ),
                                            )  # Cap at 45%
                                            self.outer_progress_callback(
                                                progress,
                                                100,
                                                f"Downloading country data... {mb_downloaded:.1f} MB",
                                            )

                    # Report download complete
                    if self.outer_progress_callback:
                        self.outer_progress_callback(50, 100, "Download complete")

                finally:
                    session.close()

                return str(output_file)

        # Download the archive
        if self._progress_callback:
            downloader = CountryDataDownloader(self._progress_callback)
            archive_path = pooch.retrieve(
                url="https://github.com/nvkelso/natural-earth-vector/archive/refs/tags/v5.1.2.zip",
                known_hash=None,
                path=self._cache_dir,
                fname="natural-earth-v5.1.2.zip",
                downloader=downloader,
            )
        else:
            archive_path = pooch.retrieve(
                url="https://github.com/nvkelso/natural-earth-vector/archive/refs/tags/v5.1.2.zip",
                known_hash=None,
                path=self._cache_dir,
                fname="natural-earth-v5.1.2.zip",
            )

        # Extract the specific GeoJSON file we need
        if self._progress_callback:
            self._progress_callback(50, 100, "Extracting country data...")

        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # Extract only the file we need
            zip_ref.extract(
                "natural-earth-vector-5.1.2/geojson/ne_110m_admin_0_countries.geojson",
                extract_dir,
            )

        if self._progress_callback:
            self._progress_callback(100, 100, "Country data ready")

        return geojson_path

    def _load_countries_data(self) -> gpd.GeoDataFrame:
        """Load countries data from Natural Earth GeoJSON."""
        if self._countries_gdf is None:
            geojson_path = self._get_countries_data_path()
            self._countries_gdf = gpd.read_file(geojson_path)
        return self._countries_gdf

    def _build_name_lookup(self) -> Dict[str, str]:
        """Build lookup dictionary for country name variations."""
        if self._name_lookup is not None:
            return self._name_lookup

        countries = self._load_countries_data()
        lookup = {}

        for _, row in countries.iterrows():
            name_en = row.get("NAME_EN", "").strip()
            name_long = row.get("NAME_LONG", "").strip()
            iso_a2 = row.get("ISO_A2", "").strip()
            iso_a3 = row.get("ISO_A3", "").strip()

            if not name_en:
                continue

            # Primary name (case-insensitive)
            lookup[name_en.lower()] = name_en

            # Long name if different
            if name_long and name_long != name_en:
                lookup[name_long.lower()] = name_en

            # ISO codes
            if iso_a2 and iso_a2 != "-99":
                lookup[iso_a2.lower()] = name_en
            if iso_a3 and iso_a3 != "-99":
                lookup[iso_a3.lower()] = name_en

        # Add common aliases
        aliases = {
            "uk": "United Kingdom",
            "usa": "United States of America",
            "us": "United States of America",
            "russia": "Russia",
            "south korea": "South Korea",
            "north korea": "North Korea",
        }

        for alias, canonical in aliases.items():
            if canonical.lower() in lookup:
                lookup[alias.lower()] = lookup[canonical.lower()]

        self._name_lookup = lookup
        return lookup

    def _resolve_country_name(self, country_name: str) -> str:
        """Resolve country name to canonical form."""
        lookup = self._build_name_lookup()
        normalized = country_name.strip().lower()

        # Direct lookup
        if normalized in lookup:
            return lookup[normalized]

        # Fuzzy matching
        matches = difflib.get_close_matches(normalized, lookup.keys(), n=1, cutoff=0.8)

        if matches:
            return lookup[matches[0]]

        raise ValueError(
            f"Country '{country_name}' not found. Use list_countries() to see available options."
        )

    def get_bbox(self, country_name: str) -> Tuple[float, float, float, float]:
        """Get bounding box for country as (west, south, east, north)."""
        canonical_name = self._resolve_country_name(country_name)
        countries = self._load_countries_data()

        country_row = countries[countries["NAME_EN"] == canonical_name]
        if country_row.empty:
            raise ValueError(f"Country '{canonical_name}' not found in dataset")

        bounds = country_row.iloc[0].geometry.bounds
        return bounds  # (west, south, east, north)

    def get_geometry(self, country_name: str) -> gpd.GeoDataFrame:
        """Get full country geometry for precise tile intersection."""
        canonical_name = self._resolve_country_name(country_name)
        countries = self._load_countries_data()

        country_gdf = countries[countries["NAME_EN"] == canonical_name].copy()
        if country_gdf.empty:
            raise ValueError(f"Country '{canonical_name}' not found in dataset")

        return country_gdf

    def list_countries(self) -> List[str]:
        """List all available country names."""
        countries = self._load_countries_data()
        return sorted(countries["NAME_EN"].dropna().tolist())

    def search_countries(self, query: str) -> List[str]:
        """Fuzzy search for country names."""
        all_countries = self.list_countries()
        matches = difflib.get_close_matches(
            query.lower(), [name.lower() for name in all_countries], n=10, cutoff=0.3
        )

        # Return original case versions
        result = []
        for match in matches:
            for country in all_countries:
                if country.lower() == match:
                    result.append(country)
                    break

        return result


# Global instance for convenience
_country_lookup = None


def get_country_lookup(progress_callback: Optional[Callable] = None) -> CountryLookup:
    """Get global CountryLookup instance.

    Args:
        progress_callback: Optional callback for progress updates when downloading data
    """
    global _country_lookup
    if _country_lookup is None or progress_callback is not None:
        _country_lookup = CountryLookup(progress_callback=progress_callback)
    return _country_lookup


def get_country_bbox(
    country_name: str, progress_callback: Optional[Callable] = None
) -> Tuple[float, float, float, float]:
    """Simple function to get country bounding box.

    Args:
        country_name: Name of the country
        progress_callback: Optional callback for progress updates when downloading data
    """
    return get_country_lookup(progress_callback).get_bbox(country_name)


def get_country_tiles(country_name: str, year: int = 2024) -> List[Tuple[float, float]]:
    """Get list of GeoTessera tile coordinates that intersect with country."""
    from .core import GeoTessera

    # Get country bounding box
    country_lookup = get_country_lookup()
    bbox = country_lookup.get_bbox(country_name)

    # Use existing registry to find tiles in the bounding box
    gt = GeoTessera()
    tiles = gt.registry.load_blocks_for_region(bbox, year)

    return tiles
