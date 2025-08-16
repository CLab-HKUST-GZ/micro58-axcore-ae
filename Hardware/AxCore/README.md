# AxCore: Hardware Design & Functional Verification


## Project Overview
This document outlines the hardware design and functional verification process for the AxCore project. The core is implemented using [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/master/index.html), a modern, high-level hardware description language that facilitates efficient and flexible hardware design.



## Directory Structure
All project files are organized as follows:

* **SpinalHDL Source Code:** `hw/spinal/AxCore`
* **Verification & Testbenches:** `hw/spinal/AxCore/Testing`
* **Generated Verilog RTL:** `hw/gen/AxCore/` (This directory is created after running the generator)



## Environment Setup
### Recommended: Pre-configured Environment

We have prepared a ready-to-use development environment for you, which can be accessed via docker pull.

```bash
docker pull victorchan433/axcore-dev

docker run --rm -it victorchan433/axcore-dev /bin/bash

# After entered
cd /workspace
git clone https://github.com/CLab-HKUST-GZ/micro58-axcore-ae.git

# Navigate to the project's root directory
cd micro58-axcore-ae/Hardware/AxCore/
```



## Generating Verilog RTL

Follow these steps to generate the Verilog RTL from the SpinalHDL source code. The process uses SBT (Simple Build Tool) to compile the Scala-based SpinalHDL code and execute the generator.

```bash
# Launch the SBT (Simple Build Tool) interactive shell
cs launch sbt

# Within the sbt shell, compile the project's source code
compile

# Generate Verilog files for W4 configuration of the systolic array
runMain AxCore.SystolicArray_W4.AxCore_SharedAdd_MPWq4_SA_Gen

# Generate Verilog files for W8 configuration of the systolic array
runMain AxCore.SystolicArray_W8.AxCore_SharedAdd_W8_SA_Gen

# Wait for the "[success]" message, which indicates completion, then exit the sbt shell
exit
```
The generated Verilog files can be found in the following output directory:
`hw/gen/AxCore/`.



## Functional Verification with Iverilog

To run the functional verification suite, you'll use the SBT environment to launch simulations. The testbench will report the DUT (Device Under Test) results and log a comparison against a golden reference model directly to your terminal.

### Running the Complete Test Suite
This command executes all verification tests in sequence.

```bash
# Make sure you're still in the project's root directory
cd /workspace/micro58-axcore-ae/Hardware/AxCore/

# Launch the SBT interactive shell
cs launch sbt

# (Optional) If you've modified the source code, re-compile the project first
compile

# Run the complete functional test suite
runMain AxCore.Testing.OverallFunctionalTest
```

### Running the Complete Test Suite
The `OverallFunctionalTest` suite is composed of several independent test modules. You can also execute these tests individually for more targeted debugging. Make sure you are inside the SBT shell before running these commands.

```bash
# 1. Test for Subnormal Number Conversion (SNC)
# Verifies the logic for handling subnormal floating-point numbers.
runMain AxCore.Testing.TestCases.Test_SNC_W4

# 2. Test for Mixed-Precision Floating-Point Multiplication Approximation (mpFPMA)
# Verifies the correctness of the approximate multiplication unit.
runMain AxCore.Testing.TestCases.Test_mpFPMA

# 3. Test for the 4x4 Systolic Array
# Verifies the core functionality of the systolic array dataflow and computation.
runMain AxCore.Testing.TestCases.Test_SA_4x4
```