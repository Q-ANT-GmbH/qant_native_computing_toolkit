# 3. implement manual dlpack

Date: 2025-03-24

## Status

Accepted

## Context

We want to provide a C interface to the toolkit that is as standardised as possible to ease the adoption by users.
For sharing of tensor data, [dlpack](https://github.com/dmlc/dlpack) is the standard format.
For example, torch implements a [converter](https://github.com/pytorch/pytorch/blob/d2c0c65ea1fae888ece9ffba7d2753179b2be8cc/aten/src/ATen/DLConvertor.cpp#L320) from dlpack to its internal data structure.
Dlpack itself only offers a C header but no rust bindings.
There are a bunch of crates that implement a dlpack tensor in rust but they are poorly maintained (dlpack itself is 8 years old and so are most of the crates).
The definition of the Dlpack interface is relatively simple though, so we can mimic it in rust ourselves.

## Decision

We implement our own rust bindings for a dlpack compatible data type.

## Consequences

We have to watch out for changes in the dlpack specifications and users that update to new standards.
These changes are very rare, with substantial modifications required only every few years, and not all users immediately update (e.g., torch has not updated for 3 years at least).
We can, through testing against a C++ application that uses the official dlpack.h, make sure that the rust bindings are actually compliant.
