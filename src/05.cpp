#include "matrix.hpp"
#include "time.hpp"

#ifndef USE_DOUBLE
using real_type = float;
#else
using real_type = double;
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
            e = q.submit([&](sycl::handler &cgh) {
                auto cache = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);

                cgh.parallel_for(sycl::nd_range{ sycl::range(WIDTH, (Ndiv - 1 - offset) * WIDTH), sycl::range(WIDTH, WIDTH) }, [=](sycl::nd_item<2> idx) {
                    const auto g = idx.get_group();

                    const auto groupj = g.get_group_id(1);
                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    const auto offsetj = groupj + 1;

                    const auto idxgroup = [=](size_t i, size_t j) -> size_t {
                        return index(i + offset * WIDTH, j + offset * WIDTH);
                    };

                    // Cholesky

                    cholesky_decomposition<WIDTH>(gpuA, jitter, cache, g, idxgroup);
                    sycl::group_barrier(idx.get_group());

                    // Lower Column

                    auto x = gpuA[idxgroup(locali, localj + offsetj * WIDTH)];
                    const auto diag = cache[locali][locali];

                    for (size_t k = 0; k < WIDTH - 1; ++k) {
                        if (locali == k) {
                            cache[localj][localj] = x /= diag;
                        }

                        sycl::group_barrier(g);

                        if (locali > k) {
                            x -= cache[k][locali] * cache[localj][localj];
                        }

                        sycl::group_barrier(g);
                    }

                    if (locali == WIDTH - 1) {
                        x /= diag;
                    }

                    gpuA[idxgroup(locali, localj + offsetj * WIDTH)] = x;
                });
            });
            times.push_back(getduration(e));

            e = q.submit([&](sycl::handler &cgh) {
                auto cacheline = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);
                auto cacheblock = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{ sycl::range((Ndiv - 1 - offset) * WIDTH, (Ndiv - 1 - offset) * WIDTH), sycl::range(WIDTH, WIDTH) }, [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        if (groupi > groupj) {
                            return;
                        }

                        const auto idxgroup = [=](size_t i, size_t j) -> size_t {
                            return index(i + offset * WIDTH, j + offset * WIDTH);
                        };

                        const auto offseti = groupi + 1;
                        const auto offsetj = groupj + 1;

                        cacheblock[locali][localj] = gpuA[idxgroup(locali, localj + offsetj * WIDTH)];
                        cacheline[locali][localj] = gpuA[idxgroup(locali, localj + offseti * WIDTH)];

                        sycl::group_barrier(idx.get_group());

                        real_type sum = 0;

                        for (size_t k = 0; k < WIDTH; ++k) {
                            sum += cacheblock[k][localj] * cacheline[k][locali];
                        }

                        gpuA[idxgroup(locali + offseti * WIDTH, localj + offsetj * WIDTH)] -= sum;
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

    benchmarkCholesky<real_type>("05", true, false, q, idx, gen, kernel);
    // for (size_t N = 1; N <= 1024; ++N) {
    //     validateCholesky<real_type>(false, q, N, idx, gen, kernel);
    // }

    return 0;
}
