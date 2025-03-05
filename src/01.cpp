#include "matrix.hpp"
#include "time.hpp"

#ifndef USE_DOUBLE
using real_type = float;
#else
using real_type = double;
#endif

#include "benchmark.hpp"
// #include "validate.hpp"

int main(int argc, char **argv) {
    sycl::queue q(sycl::gpu_selector_v, { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() });

    const auto kernel = [&q](size_t N, real_type *gpuA, real_type jitter) {
        std::vector<uint64_t> times;
        times.reserve((N - 1) * 2 + 1);

        sycl::event e;

        for (size_t k = 0; k < N; ++k) {
            e = q.single_task(e, [=] {
                gpuA[row_idx(k, k)] = sycl::sqrt(gpuA[row_idx(k, k)] + jitter);
                for (size_t i = k + 1; i < N; ++i) {
                    gpuA[row_idx(i, k)] /= gpuA[row_idx(k, k)];
                }
            });
            times.push_back(getduration(e));

            if (k < N - 1) {
                e = q.parallel_for(sycl::range<1>(N - (k + 1)), e, [=](sycl::id<1> idx) {
                    const auto j = idx[0] + k + 1;
                    for (size_t i = j; i < N; ++i) {
                        gpuA[row_idx(i, j)] -= gpuA[row_idx(i, k)] * gpuA[row_idx(j, k)];
                    }
                });
                times.push_back(getduration(e));
            }
        }

        return times;
    };

    const auto idx = [](size_t N, size_t i, size_t j) { return row_idx(i, j); };

    benchmarkCholesky<real_type>("01", true, false, q, idx, row_idx_alloc, kernel);
    // for (size_t N = 8196; N <= 8196; ++N) {
    //     validateCholesky<real_type>(false, q, N, idx, row_idx_alloc, kernel);
    // }

    return 0;
}
