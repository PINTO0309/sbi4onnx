#!/bin/bash

init_str="$1"

# ディレクトリを指定する
directory="."

# 指定したディレクトリ内の全てのファイルに対してループを行う
for file in "$directory"/*
do
    # ファイルが存在する場合のみコマンドを実行する
    if [ -f "$file" ]; then
        echo @@@@@@@@@@@@@@ "$file"
        # onnxsim "$file" "$file"
        new_file="${file/_1x3x/_Nx3x}"
        sbi4onnx \
        --input_onnx_file_path "$file" \
        --output_onnx_file_path "$new_file" \
        --initialization_character_string $init_str
    fi
done
