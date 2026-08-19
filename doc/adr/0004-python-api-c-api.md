# 4. Python API != C API - Dimension Handling

Date: 2025-08-20

## Status

Accepted

## Context

The Q.ANT Native Computing Toolkit provides two primary interfaces: a Python API and a C/C++ API. A Rust interface is also possible, but not supported.

Due to significant differences in programming paradigms between Python and C/C++, the requirements and expectations for each interface differ accordingly. The Python API prioritizes ease of use and user experience, while the C/C++ API focuses on performance and strict control.

## Decision

We have decided to maintain a consistent overall structure between the Python and C/C++ APIs. However, we accept that there will be differences in how certain features are handled—particularly when those differences align with the idiomatic use of each language.

It does not make sense to invest significant effort into implementing features in a language where they are not idiomatic or necessary.

## Consequences
One area directly affected by this decision is dimension handling.

In the Python API, we aim to mimic the behavior of libraries like PyTorch, which allow for flexible handling of additional dimensions. This aligns with Python's focus on flexibility and rapid prototyping.

In the C/C++ API, we will enforce stricter dimension definitions, which are better suited for the typical performance-critical use cases and memory layout expectations of C/C++ developers.

This decision allows each interface to remain idiomatic and user-friendly in its own context, while avoiding unnecessary complexity or inefficiency.
