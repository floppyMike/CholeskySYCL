import argparse
import numpy as np
import struct

SEED = 123456789
rng = np.random.default_rng(SEED)

datatype = {
    'float': np.float32,
    'double': np.float64,
}

def main():
    parser = argparse.ArgumentParser(description='Generate vector for substitution benchmarks.')
    parser.add_argument('amount', type=int, help='Number of matrices to generate')
    parser.add_argument('datatype', choices=datatype.keys(), help='Number type')
    parser.add_argument('output_file', type=str, help='Output binary file name')
    parser.add_argument('-p', '--print', action='store_true', help='Print generated matrix')
    args = parser.parse_args()

    output_file = args.output_file
    numtype = datatype[args.datatype]

    for i in range(args.amount):
        print("Generating:", i)

        with open(f"{output_file}-{i}-output.bin", 'rb') as f:
            N = np.frombuffer(f.read(8), dtype=np.int64)[0]
            Lflat = np.frombuffer(f.read(), dtype=numtype)

        L = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1):
                L[i, j] = Lflat[i * (i + 1) // 2 + j]

        X = np.random.uniform(-1, 1, (N, 32)).astype(numtype)
        B = L @ X
        Bt = L.T.astype(numtype) @ X

        if args.print:
            print(f"L:\n{L}")
            print(f"X:\n{X}")
            print(f"B:\n{B}")
            print(f"Bt:\n{Bt}")
        else:
            with open(f"{output_file}-{i}-output-X.bin", 'wb') as f:
                f.write(struct.pack('Q', N))
                f.write(X.ravel().astype(numtype).tobytes())
            with open(f"{output_file}-{i}-output-B.bin", 'wb') as f:
                f.write(struct.pack('Q', N))
                f.write(B.ravel().astype(numtype).tobytes())
            with open(f"{output_file}-{i}-output-Bt.bin", 'wb') as f:
                f.write(struct.pack('Q', N))
                f.write(Bt.ravel().astype(numtype).tobytes())

if __name__ == '__main__':
    main()
