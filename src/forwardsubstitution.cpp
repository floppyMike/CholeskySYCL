#include "kernels.hpp"
#include "matrix.hpp"
#include "time.hpp"

#ifndef USE_DOUBLE
using real_type = float;
constexpr static size_t THREADCOLHEIGHT = 1;
constexpr static size_t THREAD_L_LEN = 10;
constexpr static size_t THREAD_B_LEN = 1;
#else
using real_type = double;
constexpr static size_t THREADCOLHEIGHT = 1;
constexpr static size_t THREAD_L_LEN = 5;
constexpr static size_t THREAD_B_LEN = 1;
#endif

#include "benchmark.hpp"
// #include "validate.hpp"

int main(int argc, char **argv) {
    sycl::queue q(sycl::gpu_selector_v, { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() });

    constexpr size_t WIDTH = 32;

    const auto kernel = [&q](size_t N, size_t Nb, const real_type *gpuA, real_type *gpuB) {
        const auto Ndiv = ceil_div(N, WIDTH);
        const auto Nbdiv = ceil_div(Nb, WIDTH);
        const auto Naligned = Ndiv * WIDTH;
        const auto Nbaligned = Nbdiv * WIDTH;

        std::vector<uint64_t> times;
        times.reserve((Ndiv - 1) * 2 + 3);

        const auto matindex = [=](size_t i, size_t j) -> size_t {
            return blockcol_idx<WIDTH>(Naligned, i, j);
        };

        const auto vecindex = [=](size_t i, size_t j) -> size_t {
            return blockvec_idx(Naligned, Nbaligned, i, j);
        };

        sycl::event e;

        e = q.submit([&](sycl::handler &cgh) {  // Transpose blocks
            auto blk = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH + 1), cgh);

            cgh.parallel_for(
                sycl::nd_range{
                    sycl::range(Nbdiv * WIDTH, Ndiv * WIDTH),
                    sycl::range(WIDTH, WIDTH) },
                [=](sycl::nd_item<2> idx) {
                    transpose<real_type, WIDTH>(idx, gpuB, blk);
                });
        });
        times.push_back(getduration(e));

        for (size_t offset = 0; offset < Ndiv; ++offset) {
            const auto blockwidth = Ndiv - offset;

            const auto matidxgroup = [=](size_t i, size_t j) -> size_t {
                return matindex(i + offset * WIDTH, j + offset * WIDTH);
            };

            const auto vecidxgroup = [=](size_t i, size_t j) -> size_t {
                return vecindex(i, j + offset * WIDTH);
            };

            e = q.submit([&](sycl::handler &cgh) {
                auto triagblk = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);
                auto cacheline = sycl::local_accessor<real_type, 2>(sycl::range(THREADCOLHEIGHT, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{ sycl::range(ceil_div(Nbdiv, THREADCOLHEIGHT) * WIDTH, WIDTH), sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto vecoffseti = groupi * THREADCOLHEIGHT;

                        triagblk[locali][localj] = gpuA[matidxgroup(locali, localj)];
                        sycl::group_barrier(g);

                        real_type x[THREADCOLHEIGHT];
                        for (size_t k = 0; k < THREADCOLHEIGHT; ++k) {
                            if (auto offset = vecoffseti + k; offset < Nbdiv) {
                                x[k] = gpuB[vecidxgroup(locali + offset * WIDTH, localj)];
                            }
                        }

                        solve_forwardsub<real_type, WIDTH>(g, triagblk, cacheline, x);

                        for (size_t k = 0; k < THREADCOLHEIGHT; ++k) {
                            if (auto offset = vecoffseti + k; offset < Nbdiv) {
                                gpuB[vecidxgroup(locali + offset * WIDTH, localj)] = x[k];
                            }
                        }
                    });
            });
            times.push_back(getduration(e));

            if (offset >= Ndiv - 1) {
                break;
            }

            e = q.submit([&](sycl::handler &cgh) {
                auto topblk = sycl::local_accessor<real_type, 3>(sycl::range(THREAD_B_LEN, WIDTH, WIDTH), cgh);
                auto leftblk = sycl::local_accessor<real_type, 3>(sycl::range(THREAD_L_LEN, WIDTH, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{
                        sycl::range(ceil_div(Nbdiv, THREAD_B_LEN) * WIDTH, ceil_div(blockwidth - 1, THREAD_L_LEN) * WIDTH),
                        sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto offseti = groupi * THREAD_B_LEN;
                        const auto offsetj = groupj * THREAD_L_LEN + 1;

                        for (size_t k = 0; k < THREAD_L_LEN; ++k) {
                            if (const auto offset = offsetj + k; offset < blockwidth) {
                                leftblk[k][locali][localj] = gpuA[matidxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        for (size_t k = 0; k < THREAD_B_LEN; ++k) {
                            if (const auto offset = offseti + k; offset < Nbdiv) {
                                topblk[k][locali][localj] = gpuB[vecidxgroup(locali + offset * WIDTH, localj)];
                            }
                        }

                        sycl::group_barrier(g);

                        real_type sum[THREAD_B_LEN * THREAD_L_LEN] = { 0 };
                        trans_matmul<real_type, WIDTH, THREAD_B_LEN, THREAD_L_LEN>(g, leftblk, topblk, sum);

                        for (size_t i = 0; i < THREAD_B_LEN; ++i) {
                            if (auto offsetii = offseti + i; offsetii < Nbdiv) {
                                for (size_t j = 0; j < THREAD_L_LEN; ++j) {
                                    if (auto offsetjj = offsetj + j; offsetjj < blockwidth) {
                                        gpuB[vecidxgroup(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[j * THREAD_B_LEN + i];
                                    }
                                }
                            }
                        }
                    });
            });
            times.push_back(getduration(e));
        }

        e = q.submit([&](sycl::handler &cgh) {  // Transpose blocks
            auto blk = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH + 1), cgh);

            cgh.parallel_for(
                sycl::nd_range{
                    sycl::range(Nbdiv * WIDTH, Ndiv * WIDTH),
                    sycl::range(WIDTH, WIDTH) },
                [=](sycl::nd_item<2> idx) {
                    transpose<real_type, WIDTH>(idx, gpuB, blk);
                });
        });
        times.push_back(getduration(e));

        return times;
    };

    const auto matidx = [](size_t N, size_t i, size_t j) { return blockcol_idx<WIDTH>(ceil_div(N, WIDTH) * WIDTH, j, i); };
    const auto matgen = [](size_t N) { return blockcol_idx_alloc<WIDTH>(ceil_div(N, WIDTH)); };
    const auto vecidx = [](size_t N, size_t Nb, size_t i, size_t j) { return blockvec_idx(ceil_div(N, WIDTH) * WIDTH, ceil_div(Nb, WIDTH) * WIDTH, i, j); };
    const auto vecgen = [](size_t N, size_t Nb) { return blockvec_idx_alloc<WIDTH>(ceil_div(N, WIDTH), ceil_div(Nb, WIDTH)); };

    benchmarkSubstitution<real_type>("fsub", q, matidx, matgen, vecidx, vecgen, kernel);
    // for (size_t N = 1; N <= 2048; ++N) {
    //     validateForwardSubsitution<real_type>(false, q, N, 32, matidx, matgen, vecidx, vecgen, kernel);
    // }

    return 0;
}
