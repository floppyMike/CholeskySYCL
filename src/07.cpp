#include "matrix.hpp"
#include "time.hpp"

#ifndef USE_DOUBLE
using real_type = float;
constexpr size_t THREADCOLWIDTH = 10;
constexpr size_t THREADSUBWIDTH = 5;
constexpr size_t THREADSUBHEIGHT = 5;
#else
using real_type = double;
constexpr size_t THREADCOLWIDTH = 10;
constexpr size_t THREADSUBWIDTH = 5;
constexpr size_t THREADSUBHEIGHT = 5;
#endif

#include "benchmark.hpp"
// #include "validate.hpp"

template <size_t WIDTH, typename IDX>
void cholesky_decomposition(real_type *gpuA, real_type jitter, sycl::local_accessor<real_type, 2> cache, sycl::group<2> g, IDX idx) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    cache[locali][localj] = gpuA[idx(locali, localj)];

    sycl::group_barrier(g);

    for (size_t k = 0; k < WIDTH - 1; ++k) {
        if (locali > k) {
            cache[locali][localj] -= cache[k][locali] * cache[k][localj] / (cache[k][k] + jitter);
        }

        sycl::group_barrier(g);
    }

    const auto diag = sycl::sqrt(cache[locali][locali] + jitter);

    sycl::group_barrier(g);

    cache[locali][localj] = (locali == localj) * diag + (locali != localj) * cache[locali][localj] / diag;
}

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

            e = q.submit([&](sycl::handler &cgh) {
                auto cacheblock = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);
                auto cacheline = sycl::local_accessor<real_type, 2>(sycl::range(THREADCOLWIDTH, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{ sycl::range(WIDTH, ceil_div((blockwidth - 1) * WIDTH, THREADCOLWIDTH)), sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto offsetj = groupj * THREADCOLWIDTH + 1;

                        const auto idxgroup = [=](size_t i, size_t j) -> size_t {
                            return index(i + offset * WIDTH, j + offset * WIDTH);
                        };

                        // Cholesky

                        cholesky_decomposition<WIDTH>(gpuA, jitter, cacheblock, g, idxgroup);
                        sycl::group_barrier(idx.get_group());

                        // Lower Column

                        real_type x[THREADCOLWIDTH];
                        for (size_t k = 0; k < THREADCOLWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                x[k] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        const auto diag = cacheblock[locali][locali];

                        for (size_t k = 0; k < WIDTH - 1; ++k) {
                            if (locali == k) {
                                for (size_t u = 0; u < THREADCOLWIDTH; ++u) {
                                    cacheline[u][localj] = x[u] /= diag;
                                }
                            }

                            sycl::group_barrier(g);

                            if (locali > k) {
                                const auto chol = cacheblock[k][locali];
                                for (size_t u = 0; u < THREADCOLWIDTH; ++u) {
                                    x[u] -= chol * cacheline[u][localj];
                                }
                            }

                            sycl::group_barrier(g);
                        }

                        if (locali == WIDTH - 1) {
                            for (size_t u = 0; u < THREADCOLWIDTH; ++u) {
                                x[u] /= diag;
                            }
                        }

                        for (size_t k = 0; k < THREADCOLWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                gpuA[idxgroup(locali, localj + offset * WIDTH)] = x[k];
                            }
                        }
                    });
            });
            times.push_back(getduration(e));

            e = q.submit([&](sycl::handler &cgh) {
                auto cacheline = sycl::local_accessor<real_type, 3>(sycl::range(THREADSUBWIDTH, WIDTH, WIDTH), cgh);
                auto cacheblock = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);

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

                        const auto idxgroup = [=](size_t i, size_t j) -> size_t {
                            return index(i + offset * WIDTH, j + offset * WIDTH);
                        };

                        const auto offseti = groupi * THREADSUBHEIGHT + 1;
                        const auto offsetj = groupj * THREADSUBWIDTH + 1;

                        for (size_t k = 0; k < THREADSUBWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                cacheline[k][locali][localj] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        for (size_t i = 0; i < THREADSUBHEIGHT; ++i) {
                            if (auto offset = offseti + i; offset < blockwidth) {
                                cacheblock[locali][localj] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }

                            sycl::group_barrier(idx.get_group());

                            real_type sum[THREADSUBWIDTH] = { 0 };

                            for (size_t k = 0; k < WIDTH; ++k) {
                                real_type left = cacheblock[k][locali];

                                real_type top[THREADSUBWIDTH];
                                for (size_t u = 0; u < THREADSUBWIDTH; ++u) {
                                    top[u] = cacheline[u][k][localj];
                                }

                                for (size_t j = 0; j < THREADSUBWIDTH; ++j) {
                                    sum[j] += left * top[j];
                                }
                            }

                            if (auto offsetii = offseti + i; offsetii < blockwidth) {
                                for (size_t j = 0; j < THREADSUBWIDTH; ++j) {
                                    if (auto offsetjj = offsetj + j; offsetjj < blockwidth) {
                                        if (offsetii <= offsetjj) {
                                            gpuA[idxgroup(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[j];
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

                const auto idxgroup = [=](size_t i, size_t j) -> size_t {
                    return index(i + groupj * WIDTH, j + groupj * WIDTH);
                };

                // Cholesky

                cholesky_decomposition<WIDTH>(gpuA, jitter, cache, g, idxgroup);
                sycl::group_barrier(idx.get_group());

                // Write

                gpuA[idxgroup(locali, localj)] = cache[locali][localj];
            });
        });
        times.push_back(getduration(e));

        return times;
    };

    const auto idx = [](size_t N, size_t i, size_t j) { return blockcol_idx<WIDTH>(ceil_div(N, WIDTH) * WIDTH, j, i); };
    const auto gen = [](size_t N) { return blockcol_idx_alloc<WIDTH>(ceil_div(N, WIDTH)); };

    benchmarkCholesky<real_type>("07", true, false, q, idx, gen, kernel);
    // for (size_t N = 16384; N <= 16384; ++N) {
    //     validateCholesky<real_type>(false, q, N, idx, gen, kernel);
    // }

    return 0;
}
