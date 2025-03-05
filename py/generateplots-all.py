import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

algo_color = {
    '01': "tab:blue",
    '02': "tab:orange",
    '03': "tab:green",
    '04': "tab:red",
    '05': "tab:purple",
    '06': "tab:brown",
    '06cpu': "tab:cyan",
    '07': "tab:olive",

    'fsub': "tab:blue",
    'bsub': "tab:orange",
}

jitter_color = {
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
    0: "tab:blue",
}

datatype = {
    'float': (np.float32, 4),
    'double': (np.float64, 8),
}

fileformat = ".pdf"

def gen_error_plots(numtype):
    errorfiles = lambda algo, jitteridx, i: f'{numtype}/{numtype}-60000-{i}-input-{algo}-rand-{jitteridx}-result.bin'
    correctfiles = lambda algo, i: f'{numtype}/{numtype}-60000-{i}-output.bin'
    errorresult = lambda f: f'results/{numtype}-60000-input-error-{f}{fileformat}'
    
    def readdata(algo):
        def a(i):
            correctfile = correctfiles(algo, i)
            with open(correctfile, 'rb') as f:
                n = np.frombuffer(f.read(8), dtype=np.int64)[0]
                correctdata = np.frombuffer(f.read(), dtype=datatype[numtype][0])

            def b(jitteridx, color):
                errorfile = errorfiles(algo, jitteridx, i)
                print(f"Processing {errorfile}...")
                with open(errorfile, 'rb') as f:
                    _ = f.read(8)
                    divs = np.frombuffer(f.read(8), dtype=np.int64)[0]
                    resultdata = np.frombuffer(f.read(), dtype=datatype[numtype][0])

                diff = correctdata[:n*(n+1)//2] - resultdata[:n*(n+1)//2]
                sumtop = 0
                sumbottom = 0
                errordata = []
                for k in range(1, n + 1):
                    print("\rCalculating errors:", k, end='')
                    sumtop += np.sum(diff[(k-1)*k//2:k*(k+1)//2]**2)
                    sumbottom += np.sum(correctdata[(k-1)*k//2:k*(k+1)//2]**2)
                    errordata.append(np.sqrt(sumtop) / np.sqrt(sumbottom))

                print()
                plt.plot(np.linspace(1, n, len(errordata)), errordata, color=color)
                return divs

            return [b(jitteridx, jitter_color[jitteridx]) for jitteridx in [0, 4, 3, 2, 1]]

        divs = [a(i) for i in range(5)][0]
        return [plt.Line2D([0], [0], color=jitter_color[c], lw=2) for c in [0, 4, 3, 2, 1]], [(f"1e-{str(div).count('0')}" if div != 0 else "0") for div in divs]

    plt.figure()
    plt.xscale('log')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Frobenius Norm Error')
    plt.title('Error vs Matrix Size')
    plt.grid(True)
    lines, labels = readdata("06")
    plt.legend(lines, labels)
    plt.savefig(errorresult("plot"), bbox_inches="tight")
    plt.close()

def gen_partime_plots(numtype):
    parttimefiles = lambda algo: [[f'{numtype}/{numtype}-60000-{i}-input-{algo}-{step}-parttime.bin' for i in range(5)] for step in range(int(np.log2(60000)) + 1)] + [[f'{numtype}/{numtype}-60000-{i}-input-{algo}-wholetime.bin' for i in range(5)]]
    parttimeresult = lambda f: f'results/{numtype}-60000-input-parttime-{f}{fileformat}'
    
    def readdata(algo, process):
        filess = parttimefiles(algo)
        x = []
        y = []
        for files in filess:
            timesdatas = []
            n = 0
            for file in files:
                print(f"Processing {file}...")
                with open(file, 'rb') as f:
                    n = np.frombuffer(f.read(8), dtype=np.int64)[0]
                    timesdatas.append(np.cumsum(np.frombuffer(f.read(), dtype=np.int64) / 1000000000))

            avg = np.mean(timesdatas)
            std = np.std(timesdatas)
            x.append(n)
            y.append(avg)

        process(algo, x, y)

    def loglog(algo, x, y):
        plt.loglog(x, y, label=f"{algo}", color=algo_color[algo])

    plt.figure()
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Matrix Size')
    plt.grid(True)
    readdata("01", loglog)
    readdata("02", loglog)
    readdata("03", loglog)
    readdata("04", loglog)
    readdata("05", loglog)
    readdata("06", loglog)
    readdata("06cpu", loglog)
    readdata("07", loglog)
    plt.legend()
    plt.savefig(parttimeresult("loglog"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Matrix Size')
    plt.grid(True)
    readdata("fsub", loglog)
    readdata("bsub", loglog)
    plt.legend()
    plt.savefig(parttimeresult("sub"), bbox_inches="tight")
    plt.close()

def gen_wholetime_plots(numtype):
    wholetimefiles = lambda algo: [f'{numtype}/{numtype}-60000-{i}-input-{algo}-wholetime.bin' for i in range(5)]
    wholetimeresult = lambda f: f'results/{numtype}-60000-input-wholetime-{f}{fileformat}'

    def readdata(algo, n, process):
        timesdatas = []
        for file in wholetimefiles(algo):
            print(f"Processing {file}...")
            with open(file, 'rb') as f:
                _ = f.read(8)
                timesdatas.append(np.frombuffer(f.read(), dtype=np.int64) / 1000000000)

        stacked_timesdatas = np.stack(timesdatas)
        avg_timesdatas = np.mean(stacked_timesdatas, axis=0)
        std_timesdatas = np.std(stacked_timesdatas, axis=0)

        print(f"------------- Wholetime for {algo} -------------")
        print("Max data:", np.max(avg_timesdatas))
        print(f"------------------------------------------------")

        process(algo, n, avg_timesdatas, std_timesdatas)

    def split1(algo, n, avg_timesdatas, std_timesdatas):
        plt.plot(np.linspace(1, n, len(avg_timesdatas)), avg_timesdatas, color=algo_color[algo])

    def split2(algo, n, avg_timesdatas, std_timesdatas):
        O2 = avg_timesdatas[0::2]
        O3 = avg_timesdatas[1::2]
        plt.plot(np.linspace(1, n, len(O2)), O2, color=algo_color[algo])
        plt.plot(np.linspace(1, n, len(O3)), O3, color=algo_color[algo])

    plt.figure()
    plt.xlabel('Iterations (k)')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Iterations')
    plt.grid(True)
    readdata("02", 60000, split2)
    readdata("03", 60000, split1)
    readdata("04", 60000, split2)
    plt.legend([plt.Line2D([0], [0], color=algo_color[c], lw=2) for c in ["02", "03", "04"]], ["02", "03", "04"])
    plt.savefig(wholetimeresult("standard"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.xlabel('Iterations (k)')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Iterations')
    plt.grid(True)
    readdata("05", 1875, split2)
    readdata("06", 1875, split2)
    readdata("06cpu", 1875, split2)
    readdata("07", 1875, split2)
    plt.legend([plt.Line2D([0], [0], color=algo_color[c], lw=2) for c in ["05", "06", "06cpu", "07"]], ["05", "06", "06cpu", "07"])
    plt.savefig(wholetimeresult("blocked"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.xlabel('Iterations (k)')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Iterations')
    plt.grid(True)
    readdata("fsub", 1875, split2)
    readdata("bsub", 1875, split2)
    plt.legend([plt.Line2D([0], [0], color=algo_color[c], lw=2) for c in ["fsub", "bsub"]], ["fsub", "bsub"])
    plt.savefig(wholetimeresult("sub"), bbox_inches="tight")
    plt.close()

    def cumulative(algo, n, avg_timesdatas, std_timesdatas):
        cum = np.cumsum(avg_timesdatas)
        plt.plot(np.linspace(1, n, len(cum)), cum, label=f"{algo}", color=algo_color[algo])

    plt.figure()
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Cumulative Time (seconds)')
    plt.title('Time vs Matrix Size')
    plt.grid(True)
    readdata("02", 60000, cumulative)
    readdata("03", 60000, cumulative)
    readdata("04", 60000, cumulative)
    readdata("05", 60000, cumulative)
    readdata("06", 60000, cumulative)
    readdata("06cpu", 60000, cumulative)
    readdata("07", 60000, cumulative)
    plt.legend()
    plt.savefig(wholetimeresult("cumulative"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Cumulative Time (seconds)')
    plt.title('Time vs Matrix Size')
    plt.grid(True)
    readdata("fsub", 60000, cumulative)
    readdata("bsub", 60000, cumulative)
    plt.legend()
    plt.savefig(wholetimeresult("sub-cumulative"), bbox_inches="tight")
    plt.close()

    def cumulative_log(algo, n, avg_timesdatas, std_timesdatas):
        cum = np.cumsum(avg_timesdatas[::-1])
        plt.loglog(np.linspace(1, n, len(cum)), cum, label=f"{algo}", color=algo_color[algo])

    plt.figure()
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Cumulative Time (seconds)')
    plt.title('Time vs Matrix Size')
    plt.grid(True)
    readdata("01", 60000, cumulative_log)
    readdata("02", 60000, cumulative_log)
    readdata("03", 60000, cumulative_log)
    readdata("04", 60000, cumulative_log)
    readdata("05", 60000, cumulative_log)
    readdata("06", 60000, cumulative_log)
    readdata("06cpu", 60000, cumulative_log)
    readdata("07", 60000, cumulative_log)
    plt.legend()
    plt.savefig(wholetimeresult("cumulativelog"), bbox_inches="tight")
    plt.close()

gen_wholetime_plots("float")
gen_wholetime_plots("double")
gen_partime_plots("float")
gen_partime_plots("double")
# gen_error_plots("float")
# gen_error_plots("double")
