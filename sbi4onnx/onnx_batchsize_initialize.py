#! /usr/bin/env python

import sys
import onnx
import onnx_graphsurgeon as gs
from typing import Optional
import struct
from argparse import ArgumentParser
from onnxsim import simplify

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


def initialize(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    initialization_character_string: Optional[str] = '-1',
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """
    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path. If not specified, no ONNX file is output.\n\
        Default: ''

    initialization_character_string: Optional[str]
        String to initialize batch size. "-1" or "N" or "xxx", etc...\n
        Default: '-1'

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    changed_graph: onnx.ModelProto
        Changed onnx ModelProto.
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    if not initialization_character_string:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The initialization_character_string cannot be empty.'
        )
        sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
    try:
        onnx_graph, _ = simplify(onnx_graph)
    except:
        pass
    graph = gs.import_onnx(onnx_graph)
    graph.cleanup().toposort()
    target_model = gs.export_onnx(graph)
    target_graph = target_model.graph

    # Change batch size in input, output and value_info
    taget_nodes = \
        list(target_graph.input) + \
        list(target_graph.value_info) + \
        list(target_graph.output)

    for tensor in taget_nodes:
        if len(tensor.type.tensor_type.shape.dim)>0:
            tensor.type.tensor_type.shape.dim[0].dim_param = initialization_character_string

    # Set dynamic batch size in reshapes (-1)
    for node in  target_graph.node:
        if node.op_type != 'Reshape':
            continue
        for init in target_graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into('q', shape, 0, -1)
                init.raw_data = bytes(shape)

    # infer_shapes
    if len(target_graph.value_info) > 0:
        target_model = onnx.shape_inference.infer_shapes(target_model)

    # Save
    if output_onnx_file_path:
        onnx.save(target_model, output_onnx_file_path)

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    # Return
    return target_model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '--output_onnx_file_path',
        type=str,
        required=True,
        help='Output onnx file path.'
    )
    parser.add_argument(
        '--initialization_character_string',
        type=str,
        required=True,
        default='-1',
        help=\
            'String to initialize batch size. "-1" or "N" or "xxx", etc... \n'+
            'Default: \'-1\''
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path
    output_onnx_file_path = args.output_onnx_file_path
    initialization_character_string = args.initialization_character_string
    non_verbose = args.non_verbose

    # Load
    onnx_graph = onnx.load(input_onnx_file_path)

    # Batchsize change
    changed_graph = initialize(
        input_onnx_file_path=None,
        onnx_graph=onnx_graph,
        output_onnx_file_path=output_onnx_file_path,
        initialization_character_string=initialization_character_string,
        non_verbose=non_verbose,
    )


if __name__ == '__main__':
    main()