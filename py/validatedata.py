from scipy.linalg import cholesky
import numpy as np
import argparse
import sys
import os

datatype = {
    'float': (np.float32, 4),
    'double': (np.float64, 8),
}

def main():
    parser = argparse.ArgumentParser(description='Validate if cholesky output makes sense.')
    parser.add_argument('prefix', type=str, help='Prefix of test data file to take')
    parser.add_argument('algo', type=str, help='Algorithm to look at')
    parser.add_argument('type', choices=datatype.keys(), help='Type of vector elements')
    parser.add_argument('amount', type=int, default=None, help='Print matrix upto an amount')
    parser.add_argument('jitter', type=int, default=None, help='What jitter value file to use')
    parser.add_argument('-p', action='store_true', help='Print the resulting matricies')
    args = parser.parse_args()

    input_file = os.path.join('testdata', f'{args.prefix}-input.bin')
    result_file = os.path.join('testresults', f'{args.prefix}-input-{args.algo}-{args.jitter}-result.bin')
    output_file = os.path.join('testdata', f'{args.prefix}-output.bin')

    n = args.amount
    dt, dtsize = datatype[args.type]
    count = n*(n+1)//2

    with open(input_file, 'rb') as f:
        _ = f.read(16)
        inputdata = np.zeros((n, n))
        inputdata[np.tril_indices(n)] = np.frombuffer(f.read(dtsize * count), dtype=dt, count=count)
        inputdata += inputdata.T - np.diag(np.diag(inputdata))

    with open(result_file, 'rb') as f:
        _ = f.read(8 + dtsize)
        resultdata = np.frombuffer(f.read(dtsize * count), dtype=dt, count=count)

    with open(output_file, 'rb') as f:
        _ = f.read(16)
        outputdata = np.frombuffer(f.read(dtsize * count), dtype=dt, count=count)

    # sciresultdata = cholesky(inputdata, lower=True)[np.tril_indices(n)]
    sciresultdata = np.zeros((n,n))[np.tril_indices(n)]

    mydiff = resultdata - outputdata
    scidiff = sciresultdata - outputdata
    mysumtop = 0
    scisumtop = 0
    sumbottom = 0

    for k in range(1, n + 1):
        mysumtop += np.sum(mydiff[(k-1)*k//2:k*(k+1)//2]**2)
        scisumtop += np.sum(scidiff[(k-1)*k//2:k*(k+1)//2]**2)

        sumbottom += np.sum(outputdata[(k-1)*k//2:k*(k+1)//2]**2)

        print(f"k: {k}, result: {np.sqrt(mysumtop) / np.sqrt(sumbottom)} sci: {np.sqrt(scisumtop) / np.sqrt(sumbottom)}")

if __name__ == "__main__":
    main()
