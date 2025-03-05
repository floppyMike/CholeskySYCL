#include "benchmark.hpp"
#include "matrix.hpp"
#include <CL/sycl.hpp>

using namespace hipsycl;

using real_type = float;

template <typename real_type, typename IDX>
auto cholesky_GPU_0(sycl::queue &q, real_type *gpuA, size_t N, IDX index) {
    sycl::event e;

    for (size_t _k = 1; _k < N; ++_k) {
        const auto k = _k - 1;

        e = q.parallel_for(sycl::range<1>(N - _k), e, [=](sycl::id<1> idx) {
            const auto i = idx[0] + _k;
            gpuA[index(i, k)] /= sycl::sqrt(gpuA[index(k, k)]);
        });

        e = q.parallel_for(
            sycl::range<2>(N - _k, N - _k), e, [=](sycl::id<2> idx) {
                const auto i = idx[0] + _k;
                const auto j = idx[1] + _k;

                if (i >= j) {
                    gpuA[index(i, j)] -= gpuA[index(i, k)] * gpuA[index(j, k)];
                }
            });
    }

    e = q.parallel_for(sycl::range<1>(N), e, [=](sycl::id<1> idx) {
        const auto k = idx[0];
        gpuA[index(k, k)] = sycl::sqrt(gpuA[index(k, k)]);
    });
}

int main(int argc, char **argv) {
    constexpr size_t N = 1 << 13;

    const auto index = [](size_t i, size_t j) -> size_t { return i * (i + 1) / 2 + j; };
    auto A = generate_random_matrix<real_type>(N * (N + 1) / 2);

    benchmarkCholesky<real_type>(A, [index](sycl::queue &q, real_type *gpuA) {
        cholesky_GPU_0(q, gpuA, N, index);
    });

    return 0;
}
