# [WIP] sbi4onnx
A very simple script that only initializes the batch size of ONNX. **S**imple **B**atchsize **I**nitialization for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/sbi4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sbi4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sbi4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sbi4onnx?color=2BAF2B)](https://pypi.org/project/sbi4onnx/) [![CodeQL](https://github.com/PINTO0309/sbi4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sbi4onnx/actions?query=workflow%3ACodeQL)

# Key concept
- [ ] Initializes the ONNX batch size with the specified characters.
- [ ] This tool is not a panacea and may fail to initialize models with very complex structures. For example, there is an ONNX that contains a `Reshape` that involves a batch size, or a `Gemm` that contains a batch output other than 1 in the output result.
- [ ] A `Reshape` in a graph cannot contain more than two undefined shapes, such as `-1` or `N`. Therefore, before initializing the batch size with this tool, make sure that the Reshape does not already contain one or more `-1` dimensions. If it already contains undefined dimensions, it may be possible to successfully initialize the batch size by pre-writing the undefined dimensions of the relevant Reshape to static values using **[sam4onnx](https://github.com/PINTO0309/sam4onnx)**.
