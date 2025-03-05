import argparse
import numpy as np
import sys

datatype = {
    'float': (np.float32, 4),
    'double': (np.float64, 8),
    'long': (np.int64, 8),
}

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def main():
    parser = argparse.ArgumentParser(description='Read binary file with initial unsigned longs and vector elements.')
    parser.add_argument('input_file', help='Path to the input binary file')
    parser.add_argument('count', type=int, help='Number of initial 8-byte unsigned longs')
    parser.add_argument('type', choices=datatype.keys(), help='Type of vector elements')
    parser.add_argument('-t', action='store_true', help='Print only the initial unsigned longs')
    parser.add_argument('-n', type=positive_int, default=None, help='Print matrix upto n')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'rb') as f:
            print(np.frombuffer(f.read(args.count * 8), dtype=np.int64))
            if args.t: return

            element_fmt, element_size = datatype[args.type]
            print(np.frombuffer(f.read(args.n * element_size if args.n != None else -1), dtype=element_fmt))
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error opening file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
