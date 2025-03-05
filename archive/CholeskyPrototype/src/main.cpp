#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

//
// Simple Square Matrix Implementation
//

class SquareMatrix {
public:
  SquareMatrix(size_t n, std::vector<double> &&data)
      : m_n(n), m_data(std::move(data)) {
    assert(m_data.size() == m_n * m_n);
    assert(m_data.size() > 0);
  }

  SquareMatrix(size_t n) : SquareMatrix(n, std::vector(n * n, 0.)) {}

  SquareMatrix(const SquareMatrix &a) : m_n(a.m_n), m_data(a.m_data) {}

  auto idx(size_t i, size_t j) const -> size_t {
    const auto r = i * m_n + j;
    assert(r < m_data.size());
    return r;
  }

  auto n() const -> size_t { return m_n; }
  auto get(size_t i, size_t j) const -> double { return m_data[idx(i, j)]; }
  void set(size_t i, size_t j, double val) { m_data[idx(i, j)] = val; }

  auto transpose() const -> SquareMatrix {
    auto t = SquareMatrix(m_n);

    for (size_t i = 0; i < m_n; ++i) {
      for (size_t j = 0; j < m_n; ++j) {
        t.set(j, i, get(i, j));
      }
    }

    return t;
  }

  void print() const {
    printf("Size: %zu\n", n());

    for (size_t i = 0; i < n(); ++i) {
      for (size_t j = 0; j < n(); ++j) {
        printf("%.6f ", get(i, j));
      }

      putchar('\n');
    }
  }

private:
  size_t m_n;
  std::vector<double> m_data;
};

//
// Simple Vector Implementation
//

class Vector {
public:
  Vector(std::vector<double> &&data) : m_data(std::move(data)) {
    assert(m_data.size() > 0);
  }

  Vector(size_t n) : Vector(std::vector(std::vector(n, 0.))) {}

  Vector(const Vector &vec) : m_data(vec.m_data) {}

  auto idx(size_t i) const -> size_t {
    assert(i < m_data.size());
    return i;
  }

  auto n() const -> size_t { return m_data.size(); }
  auto get(size_t i) const -> double { return m_data[idx(i)]; }
  void set(size_t i, double val) { m_data[idx(i)] = val; }

  void print() const {
    printf("Size: %zu\n", n());

    for (size_t i = 0; i < n(); ++i) {
      printf("%.6f\n", get(i));
    }
  }

private:
  std::vector<double> m_data;
};

//
// Algorithms
//

auto choleskyBanachiewicz(const SquareMatrix &A) -> SquareMatrix {
  auto L = SquareMatrix(A.n());

  for (size_t i = 0; i < A.n(); ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double sum = 0;

      for (size_t k = 0; k < j; ++k) {
        sum += L.get(i, k) * L.get(j, k);
      }

      if (i == j) {
        L.set(i, j, std::sqrt(A.get(i, i) - sum));
      } else {
        L.set(i, j, 1. / L.get(j, j) * (A.get(i, j) - sum));
      }
    }
  }

  return L;
}

auto choleskyCrout(const SquareMatrix &A) -> SquareMatrix {
  auto L = SquareMatrix(A.n());

  for (size_t j = 0; j < A.n(); ++j) {
    double sum = 0;
    for (size_t k = 0; k < j; ++k) {
      sum += L.get(j, k) * L.get(j, k);
    }
    L.set(j, j, std::sqrt(A.get(j, j) - sum));

    for (size_t i = j + 1; i < A.n(); ++i) {
      double sum = 0;
      for (size_t k = 0; k < j; ++k) {
        sum += L.get(i, k) * L.get(j, k);
      }
      L.set(i, j, 1. / L.get(j, j) * (A.get(i, j) - sum));
    }
  }

  return L;
}

auto choleskySubmatrixInplace(const SquareMatrix &A) -> SquareMatrix {
  auto L = SquareMatrix(A);

  for (size_t _k = 1; _k < L.n(); ++_k) {
    const auto k = _k - 1;

    for (size_t x = 0; x < L.n() - _k; ++x) {
      const auto i = x + _k;
      L.set(i, k, L.get(i, k) / std::sqrt(L.get(k, k)));
    }

    for (size_t x = 0; x < L.n() - _k; ++x) {
      for (size_t y = 0; y < L.n() - _k; ++y) {
        const auto i = x + _k;
        const auto j = y + _k;
        if (i >= j) {
          L.set(i, j, L.get(i, j) - L.get(i, k) * L.get(j, k));
        }
      }
    }
  }

  for (size_t k = 0; k < L.n(); ++k) {
    L.set(k, k, std::sqrt(L.get(k, k)));
  }

  // L.set(L.n() - 1, L.n() - 1, std::sqrt(L.get(L.n() - 1, L.n() - 1)));

  return L;
}

auto choleskyColumnInplace(const SquareMatrix &A) -> SquareMatrix {
  auto L = SquareMatrix(A);

  for (size_t j = 0; j < L.n(); ++j) {

    for (size_t k = 0; k < j; ++k) {
      for (size_t i = j; i < L.n(); ++i) {
        L.set(i, j, L.get(i, j) - L.get(i, k) * L.get(j, k));
      }
    }

    L.set(j, j, std::sqrt(L.get(j, j)));

    for (size_t i = j + 1; i < L.n(); ++i) {
      L.set(i, j, L.get(i, j) / L.get(j, j));
    }
  }

  return L;
}

//
// Merge L and Linv to save space
//

void mergeLinv(SquareMatrix &L) {
  for (size_t i = 0; i < L.n(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      L.set(j, i, L.get(i, j));
    }
  }
}

//
// Solve Ax = b with L * y = b then Linv * x = y
//

auto solve(const SquareMatrix &L, const Vector &b) -> Vector {
  const auto n = b.n();

  auto y = Vector(b);

  // Forward substitution
  for (size_t _j = 1; _j < n; ++_j) {
    const auto j = _j - 1;

    for (size_t idx = 0; idx < L.n() - _j; ++idx) {
      const auto i = idx + _j;
      y.set(i, y.get(i) - L.get(i, j) * y.get(j) / L.get(j, j));
    }
  }

  for (size_t j = 0; j < L.n(); ++j) {
    y.set(j, y.get(j) / L.get(j, j));
  }

  // Backward substitution
  for (size_t j = L.n() - 1; j > 0; --j) {
    for (size_t i = 0; i < j; ++i) {
      y.set(i, y.get(i) - L.get(j, i) * y.get(j) / L.get(j, j));
    }
  }

  for (size_t j = 0; j < L.n(); ++j) {
    y.set(j, y.get(j) / L.get(j, j));
  }

  return y;
}

int main(int argc, char **argv) {
  //
  // Inputs
  //

  const auto A = SquareMatrix(3, {4, 12, -16, 12, 37, -43, -16, -43, 98});

  puts("#############################");
  puts("Input Matrix A: ");
  A.print();
  putchar('\n');

  //
  // Cholesky Banachiewicz Algorithm
  // (https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky%E2%80%93Banachiewicz_and_Cholesky%E2%80%93Crout_algorithms)
  //

  auto L_bana = choleskyBanachiewicz(A);
  mergeLinv(L_bana);

  puts("#############################");
  puts("Cholesky Banachiewicz\n");
  puts("Output Matrix L: ");
  L_bana.print();
  putchar('\n');

  //
  // Cholesky Crout Algorithm
  // (https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky%E2%80%93Banachiewicz_and_Cholesky%E2%80%93Crout_algorithms)
  //

  auto L_crout = choleskyCrout(A);
  mergeLinv(L_crout);

  puts("#############################");
  puts("Cholesky Crout\n");
  puts("Output Matrix L: ");
  L_crout.print();
  putchar('\n');

  //
  // Submatrix Algorithm
  // https://courses.grainger.illinois.edu/cs554/fa2013/notes/07_cholesky.pdf
  //

  auto L_submatrix = choleskySubmatrixInplace(A);
  mergeLinv(L_submatrix);

  puts("#############################");
  puts("Cholesky Submatrix\n");
  puts("Output Matrix L: ");
  L_submatrix.print();
  putchar('\n');

  //
  // Column Algorithm
  // https://courses.grainger.illinois.edu/cs554/fa2013/notes/07_cholesky.pdf
  //

  auto L_column = choleskyColumnInplace(A);
  mergeLinv(L_column);

  puts("#############################");
  puts("Cholesky Column\n");
  puts("Output Matrix L: ");
  L_column.print();
  putchar('\n');

  //
  // Output
  //

  const auto b = Vector({1, 2, 3});

  puts("#############################");
  puts("Input Vector b: ");
  b.print();
  putchar('\n');

  const auto x = solve(L_bana, b);

  puts("#############################");
  puts("Output Vector x: ");
  x.print();
  putchar('\n');

  return 0;
}
