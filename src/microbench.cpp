#ifndef USE_DOUBLE
using real_type = float;
#else
using real_type = double;
#endif

#include "microbench.hpp"

int main(int argc, char **argv) {
    sycl::queue q(sycl::gpu_selector_v);

    const auto kernel1 = [&q](size_t N, real_type *gpuA) {
        q.parallel_for(sycl::range<2>(N, N), [=](sycl::id<2> idx) {
            const auto i = idx[0];
            const auto j = idx[1];
            if (i > j) {
                gpuA[i * N + j] /= 2;
            }
        });
    };

    const auto kernel2 = [&q](size_t N, real_type *gpuA) {
        for (size_t i = 0; i < N; ++i) {
            q.parallel_for(sycl::range<1>(i + 1), [=](sycl::id<1> idx) {
                const auto j = idx[0];
                gpuA[i * N + j] /= 2;
            });
        }
    };

    microbench<real_type>(q, 9999, kernel1, kernel2);

    return 0;
}
