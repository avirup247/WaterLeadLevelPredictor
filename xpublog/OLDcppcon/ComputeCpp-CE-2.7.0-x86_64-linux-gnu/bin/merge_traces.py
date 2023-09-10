###############################################################################
#
#    Copyright (C) 2002-2019 Codeplay Software Limited
#    All Rights Reserved.
#
#    Codeplay's ComputeCpp
#
###############################################################################

import os
import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Merges multiple trace files")
    parser.add_argument("--files", "-f", type=str, nargs="+", help="The list of json trace files to merge")
    parser.add_argument("--output", "-o", type=str, help="Output merged json trace file")
    
    args = parser.parse_args()

    if len(args.files) <= 1:
        print("At least two files are required")
        return

    # Check the files exist
    for input_file in args.files:
        if not os.path.exists(input_file):
            print("Input file {} does not exist".format(input_file))
            return
    
    output_json = None
    for input_file in args.files:
        with open(input_file, "r") as input:
            input_json = json.load(input)

            if "traceEvents" not in input_json:
                    print("traceEvents not found in {}".format(input_file))
                    return

            if output_json is None:                
                output_json = input_json
            else:                    
                for key in input_json:
                    if isinstance(input_json[key], list):                        
                        if key in output_json:
                            # Merge the lists
                            output_json[key] += input_json[key]
                        else:
                            # Add the list
                            output_json[key] = input_json[key]
                
    if output_json is not None:
        with open(str(args.output), "w") as output:
            json.dump(output_json, output)


if __name__ == "__main__":
    main()
