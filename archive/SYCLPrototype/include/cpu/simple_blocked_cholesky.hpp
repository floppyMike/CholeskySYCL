#include <cmath>
#include <iostream>
#include <vector>

namespace sbc {

using real_type = float;

constexpr size_t N = 5;
constexpr size_t WIDTH = 3;

inline auto index(size_t i, size_t j) -> size_t {
    return N * (N + 1) / 2 - (N - j) * (N - j + 1) / 2 + i - j;
}

inline void print_matrix(std::vector<real_type> &A) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            std::cout << A[index(i, j)] << ' ';
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

inline void cholesky_decomposition(std::vector<real_type> &A, size_t n, size_t offset) {
    const auto idx = [offset](size_t i, size_t j) { return index(i + offset, j + offset); };

    for (size_t k = 0; k < n; ++k) {
        for (size_t j = k + 1; j < n; ++j) {
            for (size_t i = j; i < n; ++i) {
                A[idx(i, j)] -= A[idx(i, k)] * A[idx(j, k)] / A[idx(k, k)];
            }
        }
    }

    for (size_t k = 0; k < n; ++k) {
        A[idx(k, k)] = std::sqrt(A[idx(k, k)]);
    }

    for (size_t k = 0; k < n; ++k) {
        for (size_t i = k + 1; i < n; ++i) {
            A[idx(i, k)] /= A[idx(k, k)];
        }
    }
}

inline void matrix_mul_bottomblock(std::vector<real_type> &A, size_t offset) {
    const auto idx = [offset](size_t i, size_t j) { return index(i + offset, j + offset); };
    const auto below = N - WIDTH - offset;

    for (size_t l = 0; l < below; ++l) {
        for (size_t j = 0; j < WIDTH; ++j) {
            for (size_t i = j + 1; i < WIDTH; ++i) {
                A[idx(WIDTH + l, i)] -= A[idx(i, j)] * A[idx(WIDTH + l, j)] / A[idx(j, j)];
            }
        }

        for (size_t j = 0; j < WIDTH; ++j) {
            A[idx(WIDTH + l, j)] /= A[idx(j, j)];
        }
    }
}

inline void matrix_sub(std::vector<real_type> &A, size_t offset) {
    const auto idx = [offset](size_t i, size_t j) { return index(i + offset, j + offset); };
    const auto below = N - WIDTH - offset;

    for (size_t i = 0; i < below; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            real_type sum = 0;

            for (size_t k = 0; k < WIDTH; ++k) {
                sum += A[idx(WIDTH + i, k)] * A[idx(WIDTH + j, k)];
            }

            A[idx(i + WIDTH, j + WIDTH)] -= sum;
        }
    }
}

//
//
//

inline void blocked_cholesky(std::vector<real_type> &A) {
    size_t offset = 0;
    for (; offset + WIDTH < N; offset += WIDTH) {
        // Step 1
        cholesky_decomposition(A, WIDTH, offset);
        print_matrix(A);

        // Step 2 & 3
        matrix_mul_bottomblock(A, offset);
        print_matrix(A);

        // Step 4
        matrix_sub(A, offset);
        print_matrix(A);

        std::cout << std::endl;
    }

    cholesky_decomposition(A, N - offset, offset);
    print_matrix(A);
}

inline void run() {
    // std::vector<real_type> A = { 16, 4, 16, -4, 5, 6, -9, 33, -28, 58 };
    std::vector<real_type> A = { 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5 };
    blocked_cholesky(A);
}
}  // namespace sbc
