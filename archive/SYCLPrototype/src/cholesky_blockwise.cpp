#include "benchmark.hpp"
#include "matrix.hpp"
#include <CL/sycl.hpp>

using namespace hipsycl;

using real_type = float;

template <size_t WIDTH, typename real_type, typename IDX>
auto cholesky_GPU(sycl::queue &q, real_type *gpuA, size_t N, IDX index) {
    sycl::event e;

    for (size_t _k = 1; _k < N; ++_k) {
        const auto k = _k - 1;

        e = q.parallel_for(sycl::range<1>(N - _k), e, [=](sycl::id<1> idx) {
            const auto i = idx[0] + _k;
            gpuA[index(i, k)] /= sycl::sqrt(gpuA[index(k, k)]);
        });

        const auto submatrixdim = ((N - _k) / WIDTH + ((N - _k) % WIDTH != 0));

        e = q.submit([&](sycl::handler &cgh) {
            auto localCoefi = sycl::local_accessor<real_type, 1>(sycl::range(WIDTH), cgh);
            auto localCoefj = sycl::local_accessor<real_type, 1>(sycl::range(WIDTH), cgh);

            sycl::stream out(1024, 256, cgh);

            cgh.parallel_for(
                sycl::nd_range{ sycl::range(submatrixdim * WIDTH, submatrixdim * WIDTH), sycl::range(WIDTH, WIDTH) }, e, [=](sycl::nd_item<2> idx) {
                    const auto groupi = idx.get_group(0);
                    const auto groupj = idx.get_group(1);

                    if (groupi < groupj) {
                        return;
                    }

                    const auto locali = idx.get_local_id(0);
                    const auto localj = idx.get_local_id(1);

                    const auto offseti = _k + groupi * WIDTH;
                    const auto offsetj = _k + groupj * WIDTH;

                    if (locali == 0) {
                        localCoefi[localj] = gpuA[index(offseti + localj, k)];
                        localCoefj[localj] = gpuA[index(offsetj + localj, k)];
                    }

                    sycl::group_barrier(idx.get_group());

                    gpuA[index(offseti + locali, offsetj + localj)] -= localCoefi[locali] * localCoefj[localj];
                });
        });
    }

    e = q.parallel_for(sycl::range<1>(N), e, [=](sycl::id<1> idx) {
        const auto k = idx[0];
        gpuA[index(k, k)] = sycl::sqrt(gpuA[index(k, k)]);
    });
}

int main(int argc, char **argv) {
    constexpr size_t N = 2;
    constexpr size_t WIDTH = 3;

    const auto index = [](size_t i, size_t j) -> size_t { return i * (i + 1) / 2 + i * (WIDTH - 1) + j; };
    auto A = generate_random_matrix<real_type>((N + WIDTH - 1) * (N + WIDTH) / 2 + (N + WIDTH - 1) * (WIDTH - 1));

    benchmarkCholesky<real_type>(A, [index](sycl::queue &q, real_type *gpuA) {
        cholesky_GPU<WIDTH>(q, gpuA, N, index);
    });

    return 0;
}
