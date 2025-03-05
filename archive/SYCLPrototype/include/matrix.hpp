#pragma once

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

inline auto ceil_div(size_t a, size_t b) -> size_t {
    return a / b + (a % b != 0);
}

inline auto rowIdx(size_t i, size_t j) -> size_t {
    return i * (i + 1) / 2 + j;
}

template <typename Real>
void printRowIdx(Real *A, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            std::cout << std::setw(2) << A[rowIdx(i, j)] << " ";
        }
        std::cout << std::endl;
    }
}

inline auto nearest_dividable(size_t a, size_t b) -> size_t {
    return ceil_div(a, b) * b;
}

template <size_t WIDTH, typename Real>
struct MatrixColumnBlock {
    static auto idx(size_t i, size_t j, Real *A, size_t naligned) -> Real & {
        const auto rem = naligned * (naligned + 1) / 2 - (naligned - i) * (naligned - i + 1) / 2;
        const auto mid = (i % WIDTH) * (i % WIDTH + 1) / 2;
        const auto prv = i / WIDTH * (WIDTH - 1) * WIDTH / 2;

        return A[rem + mid + prv + j - i];
    }

    static auto print(Real *A, size_t n) {
        const auto naligned = nearest_dividable(n, WIDTH);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < i; ++j) {
                std::cout << "\t";
            }
            for (size_t j = i; j < n; ++j) {
                std::cout << std::setw(7) << idx(i, j, A, naligned) << " ";
            }
            std::cout << std::endl;
        }
    }

    static auto allocSize(size_t n) -> size_t {
        const auto len = ceil_div(n, WIDTH);
        return len * (len + 1) / 2 * WIDTH * WIDTH;
    }
};

template <typename real_type>
struct ExampleData {
    template <typename IDX>
    static void fill(IDX A) {
        A(0, 0) = 16;
        A(0, 1) = 4;
        A(0, 2) = 16;
        A(0, 3) = -4;
        A(1, 1) = 5;
        A(1, 2) = 6;
        A(1, 3) = -9;
        A(2, 2) = 33;
        A(2, 3) = -28;
        A(3, 3) = 58;
    }
};

template <typename real_type>
struct RowAscendData {
    template <typename IDX>
    static void fill(size_t n, IDX A) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                A(i, j) = j + 1;
            }
        }
    }

    template <typename IDX>
    static auto validate(std::vector<real_type> &v, size_t n, IDX A) -> bool {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = j; i < n; ++i) {
                if (std::abs(A(i, j) - (real_type) 1) >= .000001) {
                    return false;
                }
            }
        }

        return true;
    }
};

template <typename Real>
auto generate_random_matrix(size_t n) -> std::vector<Real> {
    std::vector<Real> v(n);

    std::default_random_engine gen;
    std::uniform_real_distribution<Real> dist(1., 10.);
    for (auto &i : v) {
        i = dist(gen);
    }

    return v;
}

template <typename Real, typename IDX>
auto generate_random_positivedefinite(size_t N, size_t n, IDX idx) -> std::vector<Real> {
    // Generate Random Matrix

    std::vector<Real> v(N * N, 0.);

    std::default_random_engine gen;
    std::uniform_real_distribution<Real> dist(1., 10.);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            v[i * N + j] = dist(gen);
        }
    }

    // Create symmetric postive definite matrix

    std::vector<Real> u(N * N, 0.);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < N; ++k) {
                u[i * N + j] += v[j * N + k] * v[i * N + k];  // U = V*V^T
            }
        }
    }

    // Set Memory Layout

    v.resize(n);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            v[idx(i, j)] = u[i * N + j];
        }
    }

    return v;
}
