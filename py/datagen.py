import argparse
import numpy as np
import struct

SEED = 123456789
rng = np.random.default_rng(SEED)

def generate_rand_spd_matrix(N, numtype):
    L = np.zeros((N, N), dtype=numtype)
    L[np.eye(N, dtype=bool)] = rng.uniform(0.5, 2.0, size=N)
    L[np.tril(np.ones((N, N), dtype=bool), -1)] = rng.uniform(-1.0, 1.0, size=(N * (N - 1)) // 2)

    A = L @ L.T.astype(numtype)

    return L, A

datatype = {
    'float': np.float32,
    'double': np.float64,
}

def main():
    parser = argparse.ArgumentParser(description='Generate SPD matrices for Cholesky decomposition benchmarks.')
    parser.add_argument('matrix_size', type=int, help='Size of the matrix (N)')
    parser.add_argument('amount', type=int, help='Number of matrices to generate')
    parser.add_argument('datatype', choices=datatype.keys(), help='Number type')
    parser.add_argument('output_file', type=str, help='Output binary file name')
    parser.add_argument('-p', '--print', action='store_true', help='Print generated matrix')
    args = parser.parse_args()

    N = args.matrix_size
    output_file = args.output_file
    numtype = datatype[args.datatype]

    for i in range(args.amount):
        print("Generating:", i)

        L, A = generate_rand_spd_matrix(N, numtype)

        if args.print:
            print(f"L:\n{L}")
            print(f"A:\n{A}")
        else:
            with open(f"{output_file}-{i}-input.bin", 'wb') as f:
                f.write(struct.pack('Q', N))
                f.write(A[np.tril_indices(N)].astype(numtype).tobytes())
            with open(f"{output_file}-{i}-output.bin", 'wb') as f:
                f.write(struct.pack('Q', N))
                f.write(L[np.tril_indices(N)].astype(numtype).tobytes())

if __name__ == '__main__':
    main()
