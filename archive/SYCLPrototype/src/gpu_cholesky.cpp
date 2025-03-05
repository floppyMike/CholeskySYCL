#include "gpu/simple_blocked_cholesky.hpp"

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "0: Matrix Size\n";
        return -1;
    }

    const size_t N = std::strtoul(argv[1], nullptr, 10);

    sbc::run(N);

    return 0;
}
