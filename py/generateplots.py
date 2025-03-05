import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

datatype = {
    'float': (np.float32, 4),
    'double': (np.float64, 8),
}

timesize = {
    '01': lambda N: (N - 1) * 2 + 1,
    '02': lambda N: (N - 1) * 2 + 1,
    '03': lambda N: (N - 1) + 2,
    '04': lambda N: (N - 1) * 2 + 1,
    '05': lambda N: ((N + 32 - 1) // 32 - 1) * 2 + 1,
    '06': lambda N: ((N + 32 - 1) // 32 - 1) * 2 + 1,
    '07': lambda N: ((N + 32 - 1) // 32 - 1) * 2 + 1,
    'fsub': lambda N: ((N + 32 - 1) // 32 - 1) * 2 + 3,
    'bsub': lambda N: ((N + 32 - 1) // 32 - 1) * 2 + 3,
}

filetype = ['time', 'value']

parser = argparse.ArgumentParser(description='Generate plots from created data.')
parser.add_argument('prefix', type=str, help='Prefix of test data file to take')
parser.add_argument('datatype', choices=datatype.keys(), help='Type of the file elements')
parser.add_argument('filetype', choices=filetype, help='Type of file to generate/process')
parser.add_argument('algorithm', choices=timesize.keys(), help='Which algorithm result files to take')
parser.add_argument('dest', type=str, help='Destination path to place plots in')
args = parser.parse_args()

prefix = args.prefix
algo = args.algorithm
dt, dtsize = datatype[args.datatype]

for bench in range(5):
    print("Benchmark:", bench)

    if args.filetype == filetype[0]:
        wholetimefile = f'{prefix}-{bench}-input-{algo}-wholetime.bin'
        wholetimeresult = os.path.join(args.dest, f'wholetimebench-{prefix}-{bench}-{algo}.pdf')
        parttimefile = lambda i: f'{prefix}-{bench}-input-{algo}-{i}-parttime.bin'
        parttimeresult = os.path.join(args.dest, f'parttimebench-{prefix}-{bench}-{algo}.pdf')

        def extract(file):
            print(f"Processing {file} results...")

            with open(file, 'rb') as f:
                n = np.frombuffer(f.read(8), dtype=np.int64)[0]
                timesdata = np.cumsum(np.frombuffer(f.read(), dtype=np.int64) / 1000000000)

            return n, timesdata

        def extractsum(file):
            n, timesdata = extract(file)
            return n, timesdata[-1]

        plt.figure()
        n, timesdata = extract(wholetimefile)
        plt.plot(np.linspace(1, n, timesize[algo](n)), timesdata)
        plt.xlabel('Iterations (k)')
        plt.ylabel('Time (seconds)')
        plt.title('Time vs Iterations')
        plt.grid(True)
        plt.savefig(wholetimeresult, format='pdf')
        plt.close()

        plt.figure()
        ns, timesdatas = zip(*[extractsum(parttimefile(i)) for i in range(int(np.log2(n) + 1))] + [(n, timesdata[-1])])
        plt.loglog(ns, timesdatas)
        plt.xlabel('Matrix Size (N)')
        plt.ylabel('Time (seconds)')
        plt.title('Time vs Matrix Size')
        plt.grid(True)
        plt.savefig(parttimeresult, format='pdf')
        plt.close()

    elif args.filetype == filetype[1]:
        outputfile = f'{prefix}-{bench}-output.bin'
        resultfiles = [f'{prefix}-{bench}-input-{algo}-{i}-result.bin' for i in range(5)]

        errorbenchfile = os.path.join(args.dest, f'errorbench-{prefix}-{bench}-{algo}.pdf')

        print("Reading output.bin...")

        with open(outputfile, 'rb') as f:
            _ = f.read(8)
            outputdata = np.frombuffer(f.read(), dtype=dt)

        plt.figure()

        for file in resultfiles:
            print(f"Processing {file} results...")

            with open(file, 'rb') as f:
                n = np.frombuffer(f.read(8), dtype=np.int64)[0]
                div = np.frombuffer(f.read(8), dtype=np.int64)[0]
                resultdata = np.frombuffer(f.read(), dtype=dt)

            print("Calculating errors...")

            diff = resultdata[:n*(n+1)//2] - outputdata[:n*(n+1)//2]
            sumtop = 0
            sumbottom = 0
            errors = []
            for k in range(1, n + 1):
                print("\rCalculating errors:", k, end='')
                sumtop += np.sum(diff[(k-1)*k//2:k*(k+1)//2]**2)
                sumbottom += np.sum(outputdata[(k-1)*k//2:k*(k+1)//2]**2)
                errors.append(np.sqrt(sumtop) / np.sqrt(sumbottom))

            print()
            plt.plot(range(1, n + 1), errors, label=f"1e-{str(div).count('0')}" if div != 0 else "0")

        plt.xscale('log')
        plt.xlabel('Matrix Size')
        plt.ylabel('Frobenius Norm Error')
        plt.title('Error vs Matrix Size')
        plt.grid(True)
        plt.legend()
        plt.savefig(errorbenchfile, format='pdf')

