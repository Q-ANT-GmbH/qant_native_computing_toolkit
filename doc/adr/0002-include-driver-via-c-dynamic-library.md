# 2. include driver via C dynamic library

Date: 2025-01-14

## Status

Accepted

## Context

The native computing toolkit needs to include the native computing driver to run AI inference and other functionality on the NPU.
The toolkit should be open source while the driver contains intellectual property that can't be shared with the outside world.
The latter makes it impossible to go the standard route of dependency management which would be publishing the driver source code to crates.io and including it as a normal build dependency of the toolkit.
In this stage of the toolkit development, the problem arises in the CI, which for now is private, but will be public in the future.

To hide the driver code in the CI of the toolkit, there are two general options:

1. Include the driver source code in the toolkit CI via access to the private driver repo
2. Include the driver as a compiled lib which needs to be provided alongside the toolkit

Option 1 makes the build very simple, as the driver can be included as a standard dependency of the toolkit.
However, it is not straightforward to make the driver source code a secret that cannot leak out of CI.
There are probably options to do this safely, but there isn't enough experience in the team to be really sure we do not make an important mistake here.

Option 2 is much easier in terms of safety as the driver source code never enters the toolkit CI.
However, the build becomes significantly harder:

Rust libraries can be compiled to C libraries and can use C libraries, which is why we do the detour rust -> C -> rust to use the driver from the toolkit.
There are multiple options for the type of library built: rust library (``rlib``), static C (``staticlib``), dynamic C (``cdylib``).
``rlibs`` can't be used because they are only internally used by rust during compilation.
``staticlib`` didn't work because of multiply defined rust symbols stemming from building two libraries (the driver and the toolkit), which both link to rust itself.
``cdylib`` can be made to work so it is the library type of choice.

## Decision

We will build the driver as a C dynamic library and link it to the toolkit.

## Consequences

To use the toolkit fully, we always need to provide a compiled driver ``.so``.
Until we have a way to properly release and ship these lib files, the driver has to be manually installed next to the toolkit (which was the case anyway before, when the driver was a local dependency of the toolkit).
When we ship the NPU, driver, and toolkit to the customer, we need the library workflow anyway because we can't ship the driver source code.
Thus, the way the dependency is handled in the toolkit CI also serves as a test for the entire shipment workflow.
