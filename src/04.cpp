#include "matrix.hpp"
#include "time.hpp"

#ifndef USE_DOUBLE
using real_type = float;
constexpr auto WIDTH = 32 * 3;
#else
using real_type = double;
constexpr auto WIDTH = 32 * 2;
#endif

#include "benchmark.hpp"
// #include "validate.hpp"

int main(int argc, char **argv) {
    sycl::queue q(sycl::gpu_selector_v, { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() });

    const auto kernel = [&q](size_t N, real_type *gpuA, real_type jitter) {
        std::vector<uint64_t> times;
        times.reserve((N - 1) * 2 + 1);

        const auto index = [=](size_t i, size_t j) -> size_t {
            return filledcol_idx<WIDTH>(N, i, j);
        };

        sycl::event e;

        for (size_t _k = 1; _k < N; ++_k) {
            const auto k = _k - 1;

            e = q.submit([&](sycl::handler &cgh) {
                auto localCoefk = sycl::local_accessor<real_type, 1>(sycl::range(1), cgh);

                cgh.parallel_for(sycl::nd_range{ sycl::range(ceil_div(N - _k, WIDTH) * WIDTH), sycl::range(WIDTH) }, [=](sycl::nd_item<1> idx) {
                    const auto groupi = idx.get_group(0);
                    const auto locali = idx.get_local_id(0);

                    const auto offseti = _k + groupi * WIDTH;

                    if (locali == 0) {
                        localCoefk[0] = gpuA[index(k, k)];
                    }

                    sycl::group_barrier(idx.get_group());

                    gpuA[index(offseti + locali, k)] /= sycl::sqrt(localCoefk[0] + jitter);
                });
            });
            times.push_back(getduration(e));

            e = q.submit([&](sycl::handler &cgh) {
                auto localCoefj = sycl::local_accessor<real_type, 1>(sycl::range(WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{ sycl::range(ceil_div(N - _k, WIDTH) * WIDTH, ceil_div(N - _k, WIDTH)), sycl::range(WIDTH, 1) }, [=](sycl::nd_item<2> idx) {
                        const auto groupi = idx.get_group(0);
                        const auto groupj = idx.get_group(1);

                        const auto locali = idx.get_local_id(0);

                        const auto offseti = _k + groupi * WIDTH;
                        const auto offsetj = _k + groupj * WIDTH;

                        if (groupi < groupj) {
                            return;
                        }

                        const auto coefi = gpuA[index(offseti + locali, k)];
                        localCoefj[locali] = gpuA[index(offsetj + locali, k)];

                        sycl::group_barrier(idx.get_group());

                        for (size_t u = 0; u < WIDTH; ++u) {
                            gpuA[index(offseti + locali, offsetj + u)] -= coefi * localCoefj[u];
                            sycl::group_barrier(idx.get_group());
                        }
                    });
            });
            times.push_back(getduration(e));
        }

        e = q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            const auto k = idx[0];
            gpuA[index(k, k)] = sycl::sqrt(gpuA[index(k, k)] + jitter);
        });
        times.push_back(getduration(e));

        return times;
    };

    benchmarkCholesky<real_type>("04", true, false, q, filledcol_idx<WIDTH>, filledcol_idx_alloc<WIDTH>, kernel);
    // for (size_t N = 1; N <= 1024; ++N) {
    //     validateCholesky<real_type>(false, q, N, filledcol_idx<WIDTH>, filledcol_idx_alloc<WIDTH>, kernel);
    // }

    return 0;
}
