# 6. auto-generated-dlpack-bindings

Date: 2025-10-24

## Status

Accepted

## Context

Before this, we had a manual implementation of dlpack rust bindings (ADR-0003).
Although dlpack is a relatively small library, this is error-prone and hard to maintain.
Such manual maintenance effort would have been necessary to update from the deprecated `DLManagedTensor` to `DLManagedTensorVersioned`.

## Decision

The rust bindings are auto-generated using bindgen and the dlpack header.
This allows for easy and safe upgrades. To do so, we integrated dlpack as a submodule.

## Consequences

Short-term: Users of the C API might have to update their dlpack version (in their dev environment). Installing dlpack as a debian package won't work anymore as it has a very outdated version of dlpack. The toolkit itself is self-contained through the submodule.

Long-term: This change makes it much easier (and safer) to upgrade to newer versions of dlpack.
Still, to upgrade the version, we have to update the dlpack version in this repository manually. Ideally this is done/checked with every coming major release.
