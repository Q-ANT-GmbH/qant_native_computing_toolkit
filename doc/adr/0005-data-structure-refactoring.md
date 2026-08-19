# 5. Data structure refactoring

Date: 2025-10-16

## Status

Accepted

## Context

The current data structures `FeatureData` and `FilterData` are redundant and contain dummy dimensions for many operations.

## Decision

The internal design principle (holds for everything except Python wrapper (not bridge) and torch API) is to use **the lowest number of dimensions possible** that handles all cases.
We will use `QantTensor<const D: usize>` which allows to use `D`-dimensional arrays.
All functions should be typed as strongly as possible -> e.g., use `QantTensor2` instead of `QantTensor<D>` if possible.

## Consequences

C API will change.

Python API might change depending on the wrapper.
The wrapper is now responsible for all reshaping.

Examples were checked.

Pytorch backend changed.
