Zarr Store Point-Read Tests
===========================

These cover the fallbacks that decide what `GeoTesseraZarr.sample_at` and
`probe` return at tile and UTM zone seams — the coordinates people actually
type, and where the store has one-pixel gaps.

Setup
-----

  $ export TERM=dumb

Test: Seam-aware point reads
----------------------------

Everything here runs offline against a small in-memory zone Dataset laid out
like a real one:

  $ uv run python "$TESTDIR/store_check.py" | tail -1
  all checks passed
