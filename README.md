# sbi4onnx
A very simple script that only initializes the batch size of ONNX. **S**imple **B**atchsize **I**nitialization for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/sbi4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/sbi4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/sbi4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/sbi4onnx?color=2BAF2B)](https://pypi.org/project/sbi4onnx/) [![CodeQL](https://github.com/PINTO0309/sbi4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/sbi4onnx/actions?query=workflow%3ACodeQL)

# Key concept
- [x] Initializes the ONNX batch size with the specified characters.
- [x] This tool is not a panacea and may fail to initialize models with very complex structures. For example, there is an ONNX that contains a `Reshape` that involves a batch size, or a `Gemm` that contains a batch output other than 1 in the output result.
- [x] A `Reshape` in a graph cannot contain more than two undefined shapes, such as `-1` or `N` or `None` or `unk_*`. Therefore, before initializing the batch size with this tool, make sure that the `Reshape` does not already contain one or more `-1` dimensions. If it already contains undefined dimensions, it may be possible to successfully initialize the batch size by pre-writing the undefined dimensions of the relevant `Reshape` to static values using **[sam4onnx](https://github.com/PINTO0309/sam4onnx)**.

## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install --no-deps -U onnx-simplifier \
&& pip install -U sbi4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```bash
$ sbi4onnx -h

usage:
  sbi4onnx [-h]
  --input_onnx_file_path INPUT_ONNX_FILE_PATH
  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
  --initialization_character_string INITIALIZATION_CHARACTER_STRING
  [--non_verbose]

optional arguments:
  -h, --help
      show this help message and exit.

  --input_onnx_file_path INPUT_ONNX_FILE_PATH
      Input onnx file path.

  --output_onnx_file_path OUTPUT_ONNX_FILE_PATH
      Output onnx file path.

  --initialization_character_string INITIALIZATION_CHARACTER_STRING
      String to initialize batch size. "-1" or "N" or "xxx", etc...
      Default: '-1'

  --non_verbose
      Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```python
>>> from sbi4onnx import initialize
>>> help(initialize)

Help on function initialize in module sbi4onnx.onnx_batchsize_initialize:

initialize(
  input_onnx_file_path: Union[str, NoneType] = '',
  onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
  output_onnx_file_path: Union[str, NoneType] = '',
  initialization_character_string: Union[str, NoneType] = '-1',
  non_verbose: Union[bool, NoneType] = False
) -> onnx.onnx_ml_pb2.ModelProto

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path. If not specified, no ONNX file is output.
        Default: ''

    initialization_character_string: Optional[str]
        String to initialize batch size. "-1" or "N" or "xxx", etc...

        Default: '-1'

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    changed_graph: onnx.ModelProto
        Changed onnx ModelProto.
```

## 4. CLI Execution
```bash
$ sbi4onnx \
--input_onnx_file_path whenet_224x224.onnx \
--output_onnx_file_path whenet_Nx224x224.onnx \
--initialization_character_string N

$ sbi4onnx \
--input_onnx_file_path whenet_224x224.onnx \
--output_onnx_file_path whenet_Nx224x224.onnx \
--initialization_character_string -1

$ sbi4onnx \
--input_onnx_file_path whenet_224x224.onnx \
--output_onnx_file_path whenet_Nx224x224.onnx \
--initialization_character_string abcdefg
```

## 5. In-script Execution
```python
from sbi4onnx import initialize

onnx_graph = initialize(
  input_onnx_file_path="whenet_224x224.onnx",
  output_onnx_file_path="whenet_Nx224x224.onnx",
  initialization_character_string="abcdefg",
)

# or

onnx_graph = initialize(
  onnx_graph=graph,
  initialization_character_string="abcdefg",
)
```

## 6. Sample
### Before
![image](https://user-images.githubusercontent.com/33194443/166225839-3b8d6378-e76f-4139-b5d1-db547ba16d16.png)

### After
![image](https://user-images.githubusercontent.com/33194443/166225927-cb39ea2f-85f6-4fdd-afbc-78a46a2475a1.png)

## 7. Reference
1. https://github.com/onnx/onnx/blob/main/docs/Operators.md
2. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
3. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
4. https://github.com/PINTO0309/simple-onnx-processing-tools
5. https://github.com/PINTO0309/PINTO_model_zoo

## 8. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
