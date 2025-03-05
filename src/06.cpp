#include "matrix.hpp"
#include "time.hpp"
#include "kernels.hpp"

#ifndef USE_DOUBLE
using real_type = float;
constexpr static size_t THREADCOLWIDTH = 10;
constexpr static size_t THREADSUBWIDTH = 6;
constexpr static size_t THREADSUBHEIGHT = 6;
#else
using real_type = double;
constexpr static size_t THREADCOLWIDTH = 10;
constexpr static size_t THREADSUBWIDTH = 3;
constexpr static size_t THREADSUBHEIGHT = 3;
#endif

#include "benchmark.hpp"
// #include "validate.hpp"

int main(int argc, char **argv) {
    sycl::queue q(sycl::gpu_selector_v, { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() });

    constexpr size_t WIDTH = 32;

    const auto kernel = [&q](size_t N, real_type *gpuA, real_type jitter) {
        const auto Ndiv = ceil_div(N, WIDTH);
        const auto Naligned = Ndiv * WIDTH;

        std::vector<uint64_t> times;
        times.reserve((Ndiv - 1) * 2 + 1);

        const auto index = [=](size_t i, size_t j) -> size_t {
            return blockcol_idx<WIDTH>(Naligned, i, j);
        };

        sycl::event e;

        for (size_t offset = 0; offset < Ndiv - 1; ++offset) {
            const auto blockwidth = Ndiv - offset;

            const auto idxgroup = [=](size_t i, size_t j) -> size_t {
                return index(i + offset * WIDTH, j + offset * WIDTH);
            };

            e = q.submit([&](sycl::handler &cgh) {
                auto cholblk = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);
                auto prevrslt = sycl::local_accessor<real_type, 2>(sycl::range(THREADCOLWIDTH, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{ sycl::range(WIDTH, ceil_div((blockwidth - 1), THREADCOLWIDTH) * WIDTH), sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto offsetj = groupj * THREADCOLWIDTH + 1;

                        cholblk[locali][localj] = gpuA[idxgroup(locali, localj)];
                        sycl::group_barrier(idx.get_group());

                        cholesky_decomposition<real_type, WIDTH>(g, cholblk, jitter);
                        sycl::group_barrier(idx.get_group());

                        real_type x[THREADCOLWIDTH];
                        for (size_t k = 0; k < THREADCOLWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                x[k] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        solve_forwardsub<real_type, WIDTH, THREADCOLWIDTH>(g, cholblk, prevrslt, x);

                        for (size_t k = 0; k < THREADCOLWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                gpuA[idxgroup(locali, localj + offset * WIDTH)] = x[k];
                            }
                        }
                    });
            });
            times.push_back(getduration(e));

            e = q.submit([&](sycl::handler &cgh) {
                auto topblk = sycl::local_accessor<real_type, 3>(sycl::range(THREADSUBWIDTH, WIDTH, WIDTH), cgh);
                auto leftblk = sycl::local_accessor<real_type, 3>(sycl::range(THREADSUBHEIGHT, WIDTH, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{
                        sycl::range(ceil_div(blockwidth - 1, THREADSUBHEIGHT) * WIDTH, ceil_div(blockwidth - 1, THREADSUBWIDTH) * WIDTH),
                        sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        if (groupi * THREADSUBHEIGHT >= (groupj + 1) * THREADSUBWIDTH) {
                            return;
                        }

                        const auto offseti = groupi * THREADSUBHEIGHT + 1;
                        const auto offsetj = groupj * THREADSUBWIDTH + 1;

                        for (size_t k = 0; k < THREADSUBWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                topblk[k][locali][localj] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        for (size_t k = 0; k < THREADSUBHEIGHT; ++k) {
                            if (auto offset = offseti + k; offset < blockwidth) {
                                leftblk[k][locali][localj] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        sycl::group_barrier(idx.get_group());

                        real_type sum[THREADSUBHEIGHT * THREADSUBWIDTH] = { 0 };
                        trans_matmul<real_type, WIDTH, THREADSUBWIDTH, THREADSUBHEIGHT>(g, leftblk, topblk, sum);

                        for (size_t i = 0; i < THREADSUBHEIGHT; ++i) {
                            if (auto offsetii = offseti + i; offsetii < blockwidth) {
                                for (size_t j = 0; j < THREADSUBWIDTH; ++j) {
                                    if (auto offsetjj = offsetj + j; offsetjj < blockwidth) {
                                        if (offsetii <= offsetjj) {
                                            gpuA[idxgroup(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[i * THREADSUBWIDTH + j];
                                        }
                                    }
                                }
                            }
                        }
                    });
            });
            times.push_back(getduration(e));
        }

        e = q.submit([&](sycl::handler &cgh) {
            auto cache = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);

            cgh.parallel_for(sycl::nd_range{ sycl::range(WIDTH, Naligned), sycl::range(WIDTH, WIDTH) }, [=](sycl::nd_item<2> idx) {
                const auto g = idx.get_group();

                const auto groupj = g.get_group_id(1);
                const auto locali = g.get_local_id(0);
                const auto localj = g.get_local_id(1);

                cache[locali][localj] = gpuA[index(locali + groupj * WIDTH, localj + groupj * WIDTH)];
                sycl::group_barrier(idx.get_group());

                cholesky_decomposition<real_type, WIDTH>(g, cache, jitter);
                sycl::group_barrier(idx.get_group());

                gpuA[index(locali + groupj * WIDTH, localj + groupj * WIDTH)] = cache[locali][localj];
            });
        });
        times.push_back(getduration(e));

        return times;
    };

    const auto idx = [](size_t N, size_t i, size_t j) { return blockcol_idx<WIDTH>(ceil_div(N, WIDTH) * WIDTH, j, i); };
    const auto gen = [](size_t N) { return blockcol_idx_alloc<WIDTH>(ceil_div(N, WIDTH)); };

    benchmarkCholesky<real_type>("06", false, true, q, idx, gen, kernel);
    // for (size_t N = 16384; N <= 16384; ++N) {
    //     validateCholesky<real_type>(false, q, N, idx, gen, kernel);
    // }

    return 0;
}
