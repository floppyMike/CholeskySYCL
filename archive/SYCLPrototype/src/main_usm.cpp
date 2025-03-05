#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace hipsycl;

int main(int argc, char **argv) {
  constexpr size_t N = 3;

  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<double> C(N * N, 0.);

  sycl::queue q([](const sycl::device &dev) {
    if (dev.is_gpu()) {
      return 1;
    } else {
      return -1;
    }
  });

  // Init Inputs

  const auto gpuA = sycl::malloc_device<double>(A.size(), q);
  const auto gpuB = sycl::malloc_device<double>(B.size(), q);
  auto e1 = q.memcpy(gpuA, A.data(), sizeof(double) * A.size());
  auto e2 = q.memcpy(gpuB, B.data(), sizeof(double) * B.size());

  // Init Outputs

  const auto gpuC = sycl::malloc_device<double>(C.size(), q);
  auto e3 = q.memset(gpuC, 0, sizeof(double) * C.size());

  // Matrix multiply

  auto e4 =
      q.parallel_for(sycl::range<2>(N, N), {e1, e2, e3}, [=](sycl::id<2> idx) {
        const auto i = idx[0];
        const auto j = idx[1];

        double sum = 0.;
        for (size_t k = 0; k < N; ++k) {
          sum += gpuA[i * N + k] * gpuB[k * N + j];
        }

        gpuC[i * N + j] = sum;
      });

  // Copy Output to Host

  q.memcpy(C.data(), gpuC, sizeof(double) * C.size(), e4).wait_and_throw();

  // Clean up & Results

  sycl::free(gpuA, q);
  sycl::free(gpuB, q);
  sycl::free(gpuC, q);

  for (const auto i : C)
    std::cout << i << ' ';
  std::cout << std::endl;

  return 0;
}
