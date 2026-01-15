# FS-extreme-years

Codebase for analyzing a temporal suite of flood susceptibility (FS)
rasters to identify extreme wet and dry years at watershed and pixel
scales.

This repository supports uncertainty-aware, multi-decadal flood
susceptibility analysis using annual raster outputs from a
machine-learning flood model.

------------------------------------------------------------------------

## Abstract

This study presents a temporally continuous framework for identifying
extreme flood susceptibility years using annual flood susceptibility
(FS) rasters from 2000--2023. Watershed-level scores are derived from
normalized proportions of dry, wet, and very wet pixels using calibrated
byte-scale thresholds. Generalized Extreme Value (GEV) models, combined
with moving-block bootstrap resampling, define uncertainty-bounded wet
and dry extremes. Extreme years are refined using neighbor-year
expansion and evaluated with change-point detection and Mann--Kendall
trend analysis. Pixel-level envelopes aggregate spatial patterns across
extreme years, revealing increasing flood susceptibility and clustering
of wet extremes in the 2020s.

------------------------------------------------------------------------

## Overview

The workflow moves beyond static or snapshot-based flood susceptibility
mapping by:

-   Leveraging annual FS rasters derived from a trained ML model
-   Quantifying temporal extremes using Extreme Value Theory
-   Propagating uncertainty via block bootstrap methods
-   Detecting regime shifts and long-term trends
-   Producing spatial envelopes of extreme conditions at pixel scale

FS values are byte-scaled (0--100), where: - FS ≥ 38 defines wet pixels\
- FS ≥ 61 defines wet+ pixels

------------------------------------------------------------------------

## Repository Structure

    FS-extreme-years/
    │
    ├── config.json
    │
    ├── src/
    │   ├── general/
    │   │   └── plot-pixel-count-wet-dry-years.py
    │   │
    │   ├── diagnostics-validation/
    │   │
    │   ├── extreme-year-thresholds/
    │   │   └── compute-scores-extreme-thresholds.py
    │   │
    │   └── pixel-level-processing/
    │       ├── parallel-pixel-stats-2threshold.py
    │       ├── pixel-stats-2threshold.py
    │       ├── plot_wet_dry_year_frequencies.py
    │       └── validation-pixelthresholdexceedance.py

------------------------------------------------------------------------

## Key Components

### config.json

Centralized configuration file controlling thresholds, temporal ranges,
bootstrap and GEV parameters, parallel processing, and input/output
paths. This ensures reproducibility and facilitates sensitivity testing.

### Extreme Year Identification

**compute-scores-extreme-thresholds.py** - Computes annual watershed
scores - Fits GEV distributions to wet and dry score series - Quantifies
uncertainty using moving-block bootstrap - Refines extreme years via
neighbor-year expansion - Validates results using change-point
detection, Mann--Kendall trend tests, and autocorrelation diagnostics

### Pixel-Level Processing

Scripts in `pixel-level-processing/` compute pixel exceedances,
aggregate frequencies across years, generate spatial envelopes, and
validate threshold behavior.

### Visualization & Diagnostics

Includes annual wet/dry pixel counts, extreme-year frequency plots, ACF
diagnostics, and pixel-level validation outputs.

------------------------------------------------------------------------

## Intended Use

This framework supports long-term flood risk screening, identification
of emerging hydroclimatic regimes, spatial targeting of adaptation
measures, and reproducible, uncertainty-aware flood susceptibility
analysis.
