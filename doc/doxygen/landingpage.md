# Q.ANT native computing toolkit

This library provides functionality relevant for neural network inference on Q.ANT native processing units.
It also acts as a user interface for the Q.ANT native computing driver, which is a required dependency.

## Installation

### Dependencies

- Q.ANT native computing driver:

  To use the Q.ANT native computing toolkit, you need to have the Q.ANT native computing driver installed on your system.
  You can install the driver from the provided ``qant-native-computing-driver.<version>.deb`` installer file via ``apt``. 

- DLPack:

  We use the DLPack tensor structure (https://github.com/dmlc/dlpack) as an interface for our tensor operations. You need at least version 1.0 (for C++) and version 1.2 (for C). 


## Memory Layout

The NPU requires all input buffers to be aligned to a page boundary of 4096 bytes.
If the provided data is not page-aligned, the runtime will perform an additional copy, which can negatively impact performance.
To avoid this unnecessary copy, ensure that your data is already aligned before passing it to the NPU.

## API documentation

[📁 See all functions](files.html)
