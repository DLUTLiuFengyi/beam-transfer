import argparse
from segmentation.v01.process import transform_beam

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-a", "--ainput", type=str, help="pic a input path",
                             default=r"D:\pycharmprojects\beam-transfer\pic\try\1006.png")
    args_parser.add_argument("-b", "--binput", type=str, help="pic b input path",
                             default=r"D:\pycharmprojects\beam-transfer\pic\try\1008.png")
    args_parser.add_argument("-o", "--output", type=str, help="output path",
                             default=r"D:\pycharmprojects\beam-transfer\pic\out\1006_1008.png")
    args = args_parser.parse_args()
    print("a input: {}, b input: {}, output: {}".format(args.ainput, args.binput, args.output))

    transform_beam(args.ainput, args.binput, args.output)

if __name__ == "__main__":
    main()