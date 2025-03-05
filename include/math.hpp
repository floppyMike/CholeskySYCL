#pragma once

#include <cmath>

constexpr auto ceil_div(size_t a, size_t b) -> size_t {
    return a / b + (a % b != 0);
}

constexpr auto nearest_dividable(size_t a, size_t b) -> size_t {
    return ceil_div(a, b) * b;
}

template <typename T>
constexpr auto pow2(T a) -> T {
    return a * a;
}
