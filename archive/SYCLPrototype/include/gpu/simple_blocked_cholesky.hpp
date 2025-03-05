#include "benchmark.hpp"
#include "matrix.hpp"
#include <CL/sycl.hpp>
#include <cmath>

namespace sbc {

using namespace hipsycl;
using real_type = float;

template <size_t WIDTH, typename IDX>
void cholesky_decomposition(IDX A, sycl::local_accessor<real_type, 2> cache, sycl::group<2> g) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    cache[localj][locali] = A(locali, localj);

    sycl::group_barrier(g);

    for (size_t k = 0; k < WIDTH - 1; ++k) {
        if (const auto v = cache[locali][localj] - cache[k][locali] * cache[k][localj] / cache[k][k]; locali > k) {
            cache[locali][localj] = v;
        }

        sycl::group_barrier(g);
    }

    const auto diag = sycl::sqrt(cache[locali][locali]);
    cache[locali][localj] = (locali == localj) * diag + (locali != localj) * cache[locali][localj] / diag;  // TODO: Sequential loading every 32 (warp size) to avoid bank conflicts

    sycl::group_barrier(g);
}

inline void run(size_t N) {
    constexpr size_t THREAD_WIDTH = 1;
    // constexpr size_t THREAD_HEIGHT = 1;
    constexpr size_t WIDTH = 3;

    const auto Nalloc = MatrixColumnBlock<WIDTH, real_type>::allocSize(N);
    const auto Ndiv = ceil_div(N, WIDTH);
    const auto Naligned = Ndiv * WIDTH;

    // auto Adata = std::vector<real_type>(Nalloc);
    auto Adata = generate_random_matrix<real_type>(Nalloc);

    const auto A = [&](size_t i, size_t j) -> real_type & {
        return Adata[rowIdx(i, j)];
    };

    RowAscendData<real_type>::fill(N, A);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            // std::cout << std::setw(2) << Adata[i * N + j] << " ";
            std::cout << rowIdx(i, j) << " ";
        }
        std::cout << std::endl;
    }

    simpleTestCholesky<real_type>(Adata, [=](sycl::queue &q, real_type *gpuA) {
        sycl::event e;

        for (size_t offset = 0; offset < Ndiv - 1; offset += 1) {
            // Process lower column and diagonal update

            e = q.submit([&](sycl::handler &cgh) {
                auto cache = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH + 1), cgh);

                cgh.parallel_for(sycl::nd_range{ sycl::range(WIDTH, (Ndiv - 1 - offset) * WIDTH), sycl::range(WIDTH, WIDTH) }, e, [=](sycl::nd_item<2> idx) {
                    const auto A = [=](size_t i, size_t j) -> real_type & {
                        return MatrixColumnBlock<WIDTH, real_type>::idx(i + offset * WIDTH, j + offset * WIDTH, gpuA, Naligned);
                    };

                    const auto g = idx.get_group();

                    const auto groupi = g.get_group_id(0);
                    const auto groupj = g.get_group_id(1);
                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    if (groupj < groupi) {
                        return;
                    }

                    const auto offsetcolj = groupj + 1;

                    // Fill cache with Cholesky

                    cholesky_decomposition<WIDTH>(A, cache, g);

                    // Lower block update

                    const auto diag = cache[locali][locali];
                    auto x = A(locali, localj + offsetcolj * WIDTH);

                    // sycl::group_barrier(g);

                    for (size_t k = 0; k < WIDTH - 1; ++k) {
                        if (locali == k) {
                            cache[localj][localj] = x /= diag;
                        }

                        const auto p = cache[k][locali];

                        sycl::group_barrier(g);

                        x -= (locali > k) * p * cache[localj][localj];
                    }

                    A(locali, localj + offsetcolj * WIDTH) = x;
                });
            });

            // Update inner block

            e = q.submit([&](sycl::handler &cgh) {
                auto cacheline = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);
                auto cacheblock = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH), cgh);

                cgh.parallel_for(sycl::nd_range{ sycl::range((Ndiv - 1 - offset) * WIDTH, ceil_div(Ndiv - 1 - offset, THREAD_WIDTH) * WIDTH), sycl::range(WIDTH, WIDTH) }, e, [=](sycl::nd_item<2> idx) {
                    const auto g = idx.get_group();

                    const auto groupi = g.get_group_id(0);
                    const auto groupj = g.get_group_id(1);

                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    const auto offsetblocki = groupi + 1;
                    const auto offsetblockj = groupj * THREAD_WIDTH + 1;

                    if (offsetblockj < offsetblocki) {
                        return;
                    }

                    const auto A = [=](size_t i, size_t j) -> real_type & {
                        return MatrixColumnBlock<WIDTH, real_type>::idx(i + offset * WIDTH, j + offset * WIDTH, gpuA, Naligned);
                    };

                    cacheline[locali][localj] = A(locali + offsetblocki * WIDTH, localj);

                    for (size_t j = 0; j < THREAD_WIDTH; ++j) {
                        if (j + offsetblockj + offset >= Ndiv) {
                            break;
                        }

                        // cacheblock[locali][localj] =  ? A(locali + (offsetblocki + j) * WIDTH, localj);

                        sycl::group_barrier(g);

                        real_type sum = 0;

                        for (size_t k = 0; k < WIDTH; ++k) {
                            sum += cacheline[k][locali] * cacheblock[k][localj];
                        }

                        A(locali + (offsetblocki + j) * WIDTH, localj + (offsetblockj) *WIDTH) -= sum;
                    }

                    if (offsetblocki == offsetblockj) {  // On the diagonal
                        sycl::group_barrier(g);

                        real_type sum = 0;

                        for (size_t k = 0; k < WIDTH; ++k) {
                            sum += cacheline[k][locali] * cacheline[k][localj];
                        }

                        A(locali + (offsetblocki) *WIDTH, localj + (offsetblockj) *WIDTH) -= sum;

                    } else {  // Inside
                        for (size_t j = 0; j < THREAD_WIDTH; ++j) {
                            if (j + offsetblockj + offset >= Ndiv) {
                                break;
                            }

                            cacheblock[locali][localj] = A(locali + (offsetblocki + j) * WIDTH, localj);

                            sycl::group_barrier(g);

                            real_type sum = 0;

                            for (size_t k = 0; k < WIDTH; ++k) {
                                sum += cacheline[k][locali] * cacheblock[k][localj];
                            }

                            A(locali + (offsetblocki + j) * WIDTH, localj + (offsetblockj) *WIDTH) -= sum;
                        }
                    }
                });
            });
        }

        // Perform Cholesky on Diagonal

        e = q.submit([&](sycl::handler &cgh) {
            auto cache = sycl::local_accessor<real_type, 2>(sycl::range(WIDTH, WIDTH + 1), cgh);

            cgh.parallel_for(sycl::nd_range{ sycl::range(Naligned, WIDTH), sycl::range(WIDTH, WIDTH) }, e, [=](sycl::nd_item<2> idx) {
                const auto g = idx.get_group();

                const auto groupi = g.get_group_id(0);
                const auto locali = g.get_local_id(0);
                const auto localj = g.get_local_id(1);

                const auto A = [=](size_t i, size_t j) -> real_type & {
                    return gpuA[rowIdx(i + groupi * WIDTH, j + groupi * WIDTH)];
                };

                cholesky_decomposition<WIDTH>(A, cache, g);

                A(locali, localj) = cache[localj][locali];
            });
        });
    });

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << std::setw(2) << Adata[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

}  // namespace sbc
