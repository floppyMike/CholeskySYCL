#include "kernels.hpp"
#include "matrix.hpp"
#include "time.hpp"

#ifndef USE_DOUBLE
using real_type = float;
constexpr static size_t CHOL_THREADCOLWIDTH = 10;
constexpr static size_t CHOL_THREADSUBWIDTH = 6;
constexpr static size_t CHOL_THREADSUBHEIGHT = 6;
constexpr static size_t FSUB_THREADCOLHEIGHT = 1;
constexpr static size_t FSUB_THREAD_L_LEN = 10;
constexpr static size_t FSUB_THREAD_B_LEN = 1;
constexpr static size_t BSUB_THREADCOLHEIGHT = 1;
constexpr static size_t BSUB_THREAD_L_LEN = 10;
constexpr static size_t BSUB_THREAD_B_LEN = 1;
#else
using real_type = double;
constexpr static size_t CHOL_THREADCOLWIDTH = 10;
constexpr static size_t CHOL_THREADSUBWIDTH = 3;
constexpr static size_t CHOL_THREADSUBHEIGHT = 3;
constexpr static size_t FSUB_THREADCOLHEIGHT = 1;
constexpr static size_t FSUB_THREAD_L_LEN = 5;
constexpr static size_t FSUB_THREAD_B_LEN = 1;
constexpr static size_t BSUB_THREADCOLHEIGHT = 1;
constexpr static size_t BSUB_THREAD_L_LEN = 4;
constexpr static size_t BSUB_THREAD_B_LEN = 1;
#endif

// #include "benchmark.hpp"
#include "validate.hpp"

int main(int argc, char **argv) {
    sycl::queue q(sycl::gpu_selector_v, { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() });

    constexpr size_t WIDTH = 32;

    //
    // Cholesky
    //

    const auto choleskykernel = [&q](size_t N, real_type *gpuA, real_type jitter) {
        const auto Ndiv = ceil_div(N, WIDTH);
        const auto Naligned = Ndiv * WIDTH;

        std::vector<uint64_t> times;
        times.reserve((Ndiv - 1) * 2 + 1);

        const auto index = [=](size_t i, size_t j) -> size_t {
            return plssvm_idx<WIDTH>(N, i, j);
        };

        sycl::event e;

        for (size_t offset = 0; offset < Ndiv - 1; ++offset) {
            const auto blockwidth = Ndiv - offset;

            const auto idxgroup = [=](size_t i, size_t j) -> size_t {
                return index(i + offset * WIDTH, j + offset * WIDTH);
            };

            e = q.submit([&](sycl::handler &cgh) {
                auto cholblk = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);
                auto prevrslt = sycl::local_accessor<real_type, 2>(sycl::range(CHOL_THREADCOLWIDTH, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{ sycl::range(WIDTH, ceil_div((blockwidth - 1), CHOL_THREADCOLWIDTH) * WIDTH), sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto offsetj = groupj * CHOL_THREADCOLWIDTH + 1;

                        cholblk[locali][localj] = gpuA[idxgroup(locali, localj)];
                        sycl::group_barrier(idx.get_group());

                        cholesky_decomposition<real_type, WIDTH>(g, cholblk, jitter);
                        sycl::group_barrier(idx.get_group());

                        real_type x[CHOL_THREADCOLWIDTH];
                        for (size_t k = 0; k < CHOL_THREADCOLWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                x[k] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        solve_forwardsub<real_type, WIDTH, CHOL_THREADCOLWIDTH>(g, cholblk, prevrslt, x);

                        for (size_t k = 0; k < CHOL_THREADCOLWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                gpuA[idxgroup(locali, localj + offset * WIDTH)] = x[k];
                            }
                        }
                    });
            });
            times.push_back(getduration(e));

            e = q.submit([&](sycl::handler &cgh) {
                auto topblk = sycl::local_accessor<real_type, 3>(sycl::range(CHOL_THREADSUBWIDTH, WIDTH, WIDTH), cgh);
                auto leftblk = sycl::local_accessor<real_type, 3>(sycl::range(CHOL_THREADSUBHEIGHT, WIDTH, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{
                        sycl::range(ceil_div(blockwidth - 1, CHOL_THREADSUBHEIGHT) * WIDTH, ceil_div(blockwidth - 1, CHOL_THREADSUBWIDTH) * WIDTH),
                        sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        if (groupi * CHOL_THREADSUBHEIGHT >= (groupj + 1) * CHOL_THREADSUBWIDTH) {
                            return;
                        }

                        const auto offseti = groupi * CHOL_THREADSUBHEIGHT + 1;
                        const auto offsetj = groupj * CHOL_THREADSUBWIDTH + 1;

                        for (size_t k = 0; k < CHOL_THREADSUBWIDTH; ++k) {
                            if (auto offset = offsetj + k; offset < blockwidth) {
                                topblk[k][locali][localj] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        for (size_t k = 0; k < CHOL_THREADSUBHEIGHT; ++k) {
                            if (auto offset = offseti + k; offset < blockwidth) {
                                leftblk[k][locali][localj] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        sycl::group_barrier(idx.get_group());

                        real_type sum[CHOL_THREADSUBHEIGHT * CHOL_THREADSUBWIDTH] = { 0 };
                        trans_matmul<real_type, WIDTH, CHOL_THREADSUBWIDTH, CHOL_THREADSUBHEIGHT>(g, leftblk, topblk, sum);

                        for (size_t i = 0; i < CHOL_THREADSUBHEIGHT; ++i) {
                            if (auto offsetii = offseti + i; offsetii < blockwidth) {
                                for (size_t j = 0; j < CHOL_THREADSUBWIDTH; ++j) {
                                    if (auto offsetjj = offsetj + j; offsetjj < blockwidth) {
                                        if (offsetii <= offsetjj) {
                                            gpuA[idxgroup(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[i * CHOL_THREADSUBWIDTH + j];
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

    //
    // Forward Substitution
    //

    const auto fsubkernel = [&q](size_t N, size_t Nb, const real_type *gpuA, real_type *gpuB) {
        const auto Ndiv = ceil_div(N, WIDTH);
        const auto Nbdiv = ceil_div(Nb, WIDTH);
        const auto Naligned = Ndiv * WIDTH;
        const auto Nbaligned = Nbdiv * WIDTH;

        std::vector<uint64_t> times;
        times.reserve((Ndiv - 1) * 2 + 3);

        const auto matindex = [=](size_t i, size_t j) -> size_t {
            return plssvm_idx<WIDTH>(N, i, j);
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
                auto cacheline = sycl::local_accessor<real_type, 2>(sycl::range(FSUB_THREADCOLHEIGHT, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{ sycl::range(ceil_div(Nbdiv, FSUB_THREADCOLHEIGHT) * WIDTH, WIDTH), sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto vecoffseti = groupi * FSUB_THREADCOLHEIGHT;

                        triagblk[locali][localj] = gpuA[matidxgroup(locali, localj)];
                        sycl::group_barrier(g);

                        real_type x[FSUB_THREADCOLHEIGHT];
                        for (size_t k = 0; k < FSUB_THREADCOLHEIGHT; ++k) {
                            if (auto offset = vecoffseti + k; offset < Nbdiv) {
                                x[k] = gpuB[vecidxgroup(locali + offset * WIDTH, localj)];
                            }
                        }

                        solve_forwardsub<real_type, WIDTH>(g, triagblk, cacheline, x);

                        for (size_t k = 0; k < FSUB_THREADCOLHEIGHT; ++k) {
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
                auto topblk = sycl::local_accessor<real_type, 3>(sycl::range(FSUB_THREAD_B_LEN, WIDTH, WIDTH), cgh);
                auto leftblk = sycl::local_accessor<real_type, 3>(sycl::range(FSUB_THREAD_L_LEN, WIDTH, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{
                        sycl::range(ceil_div(Nbdiv, FSUB_THREAD_B_LEN) * WIDTH, ceil_div(blockwidth - 1, FSUB_THREAD_L_LEN) * WIDTH),
                        sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto offseti = groupi * FSUB_THREAD_B_LEN;
                        const auto offsetj = groupj * FSUB_THREAD_L_LEN + 1;

                        for (size_t k = 0; k < FSUB_THREAD_L_LEN; ++k) {
                            if (const auto offset = offsetj + k; offset < blockwidth) {
                                leftblk[k][locali][localj] = gpuA[matidxgroup(locali, localj + offset * WIDTH)];
                            }
                        }

                        for (size_t k = 0; k < FSUB_THREAD_B_LEN; ++k) {
                            if (const auto offset = offseti + k; offset < Nbdiv) {
                                topblk[k][locali][localj] = gpuB[vecidxgroup(locali + offset * WIDTH, localj)];
                            }
                        }

                        sycl::group_barrier(g);

                        real_type sum[FSUB_THREAD_B_LEN * FSUB_THREAD_L_LEN] = { 0 };
                        trans_matmul<real_type, WIDTH, FSUB_THREAD_B_LEN, FSUB_THREAD_L_LEN>(g, leftblk, topblk, sum);

                        for (size_t i = 0; i < FSUB_THREAD_B_LEN; ++i) {
                            if (auto offsetii = offseti + i; offsetii < Nbdiv) {
                                for (size_t j = 0; j < FSUB_THREAD_L_LEN; ++j) {
                                    if (auto offsetjj = offsetj + j; offsetjj < blockwidth) {
                                        gpuB[vecidxgroup(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[j * FSUB_THREAD_B_LEN + i];
                                    }
                                }
                            }
                        }
                    });
            });
            times.push_back(getduration(e));
        }

        return times;
    };

    //
    // Backward Substitution
    //

    const auto bsubkernel = [&q](size_t N, size_t Nb, const real_type *gpuA, real_type *gpuB) {
        const auto Ndiv = ceil_div(N, WIDTH);
        const auto Nbdiv = ceil_div(Nb, WIDTH);
        const auto Naligned = Ndiv * WIDTH;
        const auto Nbaligned = Nbdiv * WIDTH;

        std::vector<uint64_t> times;
        times.reserve((Ndiv - 1) * 2 + 3);

        const auto matindex = [=](size_t i, size_t j) -> size_t {
            return plssvm_idx<WIDTH>(N, i, j);
        };

        const auto vecindex = [=](size_t i, size_t j) -> size_t {
            return blockvec_idx(Naligned, Nbaligned, i, j);
        };

        sycl::event e;

        for (size_t _offset = 0; _offset < Ndiv; ++_offset) {
            const auto offset = Ndiv - 1 - _offset;

            e = q.submit([&](sycl::handler &cgh) {
                auto triagblk = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);
                auto prevrslt = sycl::local_accessor<real_type, 2>(sycl::range(BSUB_THREADCOLHEIGHT, WIDTH), cgh);

                cgh.parallel_for(
                    sycl::nd_range{ sycl::range(ceil_div(Nbdiv, BSUB_THREADCOLHEIGHT) * WIDTH, WIDTH), sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto vecoffseti = groupi * BSUB_THREADCOLHEIGHT;
                        const auto vecoffsetj = offset;
                        const auto matoffseti = offset;
                        const auto matoffsetj = offset;

                        {
                            const auto v = gpuA[matindex(locali + matoffseti * WIDTH, localj + matoffsetj * WIDTH)];  // Explicit load
                            triagblk[locali][localj] = localj + matoffsetj * WIDTH < N ? v : 1;                       // to make the compiler avoid branch (possibly cmov?)
                        }

                        sycl::group_barrier(g);

                        real_type x[BSUB_THREADCOLHEIGHT];
                        for (size_t k = 0; k < BSUB_THREADCOLHEIGHT; ++k) {
                            if (auto offsetii = vecoffseti + k; offsetii < Nbdiv) {
                                const auto v = gpuB[vecindex(locali + offsetii * WIDTH, localj + vecoffsetj * WIDTH)];
                                x[k] = locali + vecoffsetj * WIDTH < N ? v : 0;
                            }
                        }

                        solve_backwardsub<real_type, WIDTH>(g, triagblk, prevrslt, x);

                        for (size_t k = 0; k < BSUB_THREADCOLHEIGHT; ++k) {
                            if (auto offsetii = vecoffseti + k; offsetii < Nbdiv) {
                                gpuB[vecindex(locali + offsetii * WIDTH, localj + vecoffsetj * WIDTH)] = x[k];
                            }
                        }
                    });
            });
            times.push_back(getduration(e));

            if (_offset >= Ndiv - 1) {
                break;
            }

            e = q.submit([&](sycl::handler &cgh) {
                auto topblk = sycl::local_accessor<real_type, 3>(sycl::range(BSUB_THREAD_B_LEN, WIDTH, WIDTH), cgh);
                auto leftblk = sycl::local_accessor<real_type, 3>(sycl::range(BSUB_THREAD_L_LEN, WIDTH, WIDTH + 1), cgh);

                cgh.parallel_for(
                    sycl::nd_range{
                        sycl::range(ceil_div(Nbdiv, BSUB_THREAD_B_LEN) * WIDTH, ceil_div(offset, BSUB_THREAD_L_LEN) * WIDTH),
                        sycl::range(WIDTH, WIDTH) },
                    [=](sycl::nd_item<2> idx) {
                        const auto g = idx.get_group();

                        const auto groupi = g.get_group_id(0);
                        const auto groupj = g.get_group_id(1);
                        const auto locali = g.get_local_id(0);
                        const auto localj = g.get_local_id(1);

                        const auto offseti = groupi * BSUB_THREAD_B_LEN;
                        const auto offsetj = groupj * BSUB_THREAD_L_LEN;

                        for (size_t k = 0; k < BSUB_THREAD_L_LEN; ++k) {
                            if (const auto offsetjj = offsetj + k; offsetjj < offset) {
                                const auto x = gpuA[matindex(locali + offsetjj * WIDTH, localj + offset * WIDTH)];
                                leftblk[k][localj][locali] = localj + offset * WIDTH < N ? x : 0;  // Transpose & filter out
                            }
                        }

                        for (size_t k = 0; k < BSUB_THREAD_B_LEN; ++k) {
                            if (const auto offsetii = offseti + k; offsetii < Nbdiv) {
                                topblk[k][locali][localj] = gpuB[vecindex(locali + offsetii * WIDTH, localj + offset * WIDTH)];  // Tail already contains zeros from before
                            }
                        }

                        sycl::group_barrier(g);

                        real_type sum[BSUB_THREAD_B_LEN * BSUB_THREAD_L_LEN] = { 0 };
                        trans_matmul<real_type, WIDTH, BSUB_THREAD_B_LEN, BSUB_THREAD_L_LEN>(g, leftblk, topblk, sum);

                        for (size_t i = 0; i < BSUB_THREAD_B_LEN; ++i) {
                            if (auto offsetii = offseti + i; offsetii < Nbdiv) {
                                for (size_t j = 0; j < BSUB_THREAD_L_LEN; ++j) {
                                    if (auto offsetjj = offsetj + j; offsetjj < offset) {
                                        gpuB[vecindex(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[j * BSUB_THREAD_B_LEN + i];
                                    }
                                }
                            }
                        }
                    });
            });
            times.push_back(getduration(e));
        }

        e = q.submit([&](sycl::handler &cgh) {  // Transpose blocks
            auto cacheblock = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH + 1), cgh);

            cgh.parallel_for(sycl::nd_range{ sycl::range(Nbdiv * WIDTH, Ndiv * WIDTH), sycl::range(WIDTH, WIDTH) }, [=](sycl::nd_item<2> idx) {
                const auto globalij = idx.get_global_linear_id();
                const auto locali = idx.get_local_id(0);
                const auto localj = idx.get_local_id(1);

                cacheblock[locali][localj] = gpuB[globalij];

                sycl::group_barrier(idx.get_group());

                gpuB[globalij] = cacheblock[localj][locali];
            });
        });
        times.push_back(getduration(e));

        return times;
    };

    const auto matidx = [](size_t N, size_t i, size_t j) { return plssvm_idx<WIDTH>(N, j, i); };
    const auto matgen = [](size_t N) { return plssvm_idx_alloc<WIDTH>(N); };
    const auto vecidx = [](size_t N, size_t Nb, size_t i, size_t j) { return blockvec_idx(ceil_div(N, WIDTH) * WIDTH, ceil_div(Nb, WIDTH) * WIDTH, i, j); };
    const auto vecgen = [](size_t N, size_t Nb) { return blockvec_idx_alloc<WIDTH>(ceil_div(N, WIDTH), ceil_div(Nb, WIDTH)); };

    // benchmarkCholesky<real_type>("06", true, false, q, matidx, matgen, kernel);
    for (size_t N = 1; N <= 149; ++N) {
        validateAll<real_type>(false, q, N, 3, matidx, matgen, vecidx, vecgen, choleskykernel, fsubkernel, bsubkernel);
    }

    return 0;
}
