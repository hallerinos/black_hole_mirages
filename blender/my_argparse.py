import argparse
def get_args():
    parser = argparse.ArgumentParser()

    # get all script args
    _, all_arguments = parser.parse_known_args()
    double_dash_index = all_arguments.index('--')
    script_args = all_arguments[double_dash_index + 1: ]

    # add parser rules
    parser.add_argument('-save', help="output file")
    parser.add_argument('-scale', help="resolution rezie")
    parser.add_argument('-zscale', help="rescale z axis")
    parsed_script_args, _ = parser.parse_known_args(script_args)
    return parsed_script_args