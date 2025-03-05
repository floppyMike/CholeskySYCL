#pragma once

#include "math.hpp"
#include <iostream>

template <typename real_type, typename IDX>
void print_rowbased(const real_type *A, size_t n, IDX idx) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            std::cout << A[idx(n, i, j)] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << std::flush;
}

template <typename real_type, typename VECIDX>
void print_vectors(const real_type *B, size_t n, size_t nb, VECIDX idx) {
    for (size_t i = 0; i < nb; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << B[idx(n, nb, i, j)] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << std::flush;
}

inline auto row_idx(size_t i, size_t j) -> size_t {
    return i * (i + 1) / 2 + j;
}

inline auto row_idx_alloc(size_t N) -> size_t {
    return N * (N + 1) / 2;
}

template <size_t WIDTH>
auto blockrow_idx(size_t i, size_t j) -> size_t {
    const auto prevblocks = i / WIDTH;
    const auto previ = prevblocks * (prevblocks + 1) / 2;
    const auto prevj = (i % WIDTH) * (prevblocks + 1) * WIDTH + j;
    return previ + prevj;
}

template <size_t WIDTH>
auto blockrow_idx_alloc(size_t Ndiv) -> size_t {
    return Ndiv * (Ndiv + 1) / 2 * WIDTH * WIDTH;
}

template <size_t WIDTH>
auto blockcol_idx(size_t Naligned, size_t i, size_t j) -> size_t {
    const auto rem = Naligned * (Naligned + 1) / 2 - (Naligned - i) * (Naligned - i + 1) / 2;
    const auto mid = (i % WIDTH) * (i % WIDTH + 1) / 2;
    const auto prv = i / WIDTH * (WIDTH - 1) * WIDTH / 2;

    return rem + mid + prv + j - i;
}

template <size_t WIDTH>
auto blockcol_idx_alloc(size_t Ndiv) -> size_t {
    return Ndiv * (Ndiv + 1) / 2 * WIDTH * WIDTH;
}

template <size_t WIDTH>
auto filledcol_idx(size_t N, size_t i, size_t j) -> size_t {
    const auto s = WIDTH - 1;
    const auto t = s * 2 + N;
    return j * (2 * t - j + 1) / 2 + i + s - j;
}

template <size_t WIDTH>
auto filledcol_idx_alloc(size_t N) -> size_t {
    return (N + WIDTH - 1) * (N + WIDTH) / 2 + (N + WIDTH - 1) * (WIDTH - 1);
}

inline auto blockvec_idx(size_t Naligned, size_t Nbaligned, size_t i, size_t j) -> size_t {
    return Naligned * i + j;
}

template <size_t WIDTH>
auto blockvec_idx_alloc(size_t Ndiv, size_t Nbdiv) -> size_t {
    return Ndiv * Nbdiv * WIDTH * WIDTH;
}

template <size_t WIDTH>
auto plssvm_idx(size_t N, size_t i, size_t j) -> size_t {
    if (i >= N) {
        i = N;
        j = N - 1;
    }

    return i * (2 * (N + WIDTH) - i + 1) / 2 - i + j;
}

template <size_t WIDTH>
auto plssvm_idx_alloc(size_t N) -> size_t {
    return N * (N + 1) / 2 + WIDTH * N;
}
