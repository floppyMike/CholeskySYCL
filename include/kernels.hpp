#pragma once

#include "gpu.hpp"

template <typename real_type, size_t WIDTH>
void cholesky_decomposition(sycl::group<2> g,
                            sycl::local_accessor<real_type, 2> cholblk,
                            real_type jitter) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    for (size_t k = 0; k < WIDTH - 1; ++k) {
        if (locali > k) {
            cholblk[locali][localj] -= cholblk[k][locali] * cholblk[k][localj] / (cholblk[k][k] + jitter);
        }

        sycl::group_barrier(g);
    }

    const auto diag = sycl::sqrt(cholblk[locali][locali] + jitter);

    sycl::group_barrier(g);

    cholblk[locali][localj] = (locali == localj) * diag + (locali != localj) * cholblk[locali][localj] / diag;
}

template <typename real_type, size_t WIDTH, size_t THREADWIDTH>
void solve_forwardsub(sycl::group<2> g,
                      sycl::local_accessor<real_type, 2> triagblk,
                      sycl::local_accessor<real_type, 2> prevrslt,
                      real_type (&x)[THREADWIDTH]) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    const auto diag = triagblk[locali][locali];

    for (size_t k = 0; k < WIDTH - 1; ++k) {
        if (locali == k) {
            for (size_t u = 0; u < THREADWIDTH; ++u) {
                prevrslt[u][localj] = x[u] /= diag;
            }
        }

        sycl::group_barrier(g);

        if (locali > k) {
            const auto chol = triagblk[k][locali];
            for (size_t u = 0; u < THREADWIDTH; ++u) {
                x[u] -= chol * prevrslt[u][localj];
            }
        }

        sycl::group_barrier(g);
    }

    if (locali == WIDTH - 1) {
        for (size_t u = 0; u < THREADWIDTH; ++u) {
            x[u] /= diag;
        }
    }
}

template <typename real_type, size_t WIDTH, size_t THREADTOPWIDTH, size_t THREADLEFTHEIGHT>
void trans_matmul(sycl::group<2> g,
                  sycl::local_accessor<real_type, 3> leftblk,
                  sycl::local_accessor<real_type, 3> topblk,
                  real_type (&sum)[THREADLEFTHEIGHT * THREADTOPWIDTH]) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    for (size_t k = 0; k < WIDTH; ++k) {
        real_type left[THREADLEFTHEIGHT];
        for (size_t u = 0; u < THREADLEFTHEIGHT; ++u) {
            left[u] = leftblk[u][k][locali];
        }

        real_type top[THREADTOPWIDTH];
        for (size_t u = 0; u < THREADTOPWIDTH; ++u) {
            top[u] = topblk[u][k][localj];
        }

        for (size_t i = 0; i < THREADLEFTHEIGHT; ++i) {
            for (size_t j = 0; j < THREADTOPWIDTH; ++j) {
                sum[i * THREADTOPWIDTH + j] += left[i] * top[j];
            }
        }
    }
}

template <typename real_type, size_t WIDTH>
void transpose(sycl::nd_item<2> idx,
               real_type *mat,
               sycl::local_accessor<real_type, 2> blk) {
    const auto globalij = idx.get_global_linear_id();
    const auto locali = idx.get_local_id(0);
    const auto localj = idx.get_local_id(1);

    blk[locali][localj] = mat[globalij];

    sycl::group_barrier(idx.get_group());

    mat[globalij] = blk[localj][locali];
}

template <typename real_type, size_t WIDTH, size_t THREADWIDTH>
void solve_backwardsub(sycl::group<2> g,
                       sycl::local_accessor<real_type, 2> triagblk,
                       sycl::local_accessor<real_type, 2> prevrslt,
                       real_type (&x)[THREADWIDTH]) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    const auto diag = triagblk[locali][locali];

    for (size_t _k = 0; _k < WIDTH - 1; ++_k) {
        const auto k = WIDTH - 1 - _k;

        if (locali == k) {
            for (size_t u = 0; u < THREADWIDTH; ++u) {
                prevrslt[u][localj] = x[u] /= diag;
            }
        }

        sycl::group_barrier(g);

        if (locali < k) {
            const auto chol = triagblk[locali][k];
            for (size_t u = 0; u < THREADWIDTH; ++u) {
                x[u] -= chol * prevrslt[u][localj];
            }
        }

        sycl::group_barrier(g);
    }

    if (locali == 0) {
        for (size_t u = 0; u < THREADWIDTH; ++u) {
            x[u] /= diag;
        }
    }
}
