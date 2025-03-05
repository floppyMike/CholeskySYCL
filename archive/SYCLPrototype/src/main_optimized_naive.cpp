#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

#include "matrix.hpp"

using namespace hipsycl;

using real_type = float;

int main(int argc, char **argv) {
  constexpr size_t N = 16384;

  auto A = std::vector<real_type>(N * N);
  auto B = std::vector<real_type>(N * N);
  auto C = std::vector<real_type>(N * N);

  sycl::queue q([](const sycl::device &dev) {
    if (dev.is_gpu()) {
      return 1;
    } else {
      return -1;
    }
  });

  const auto device = q.get_device();
  std::cout << "Max Workgroup Size: "
            << device.get_info<sycl::info::device::max_work_group_size>()
            << std::endl;

  // Init Inputs

  const auto gpuA = sycl::malloc_device<real_type>(A.size(), q);
  const auto gpuB = sycl::malloc_device<real_type>(B.size(), q);
  auto e1 = q.memcpy(gpuA, A.data(), sizeof(real_type) * A.size());
  auto e2 = q.memcpy(gpuB, B.data(), sizeof(real_type) * B.size());

  // Init Outputs

  const auto gpuC = sycl::malloc_device<real_type>(C.size(), q);
  auto e3 = q.memset(gpuC, 0, sizeof(real_type) * C.size());

  // Setup benchmark

  q.wait();
  const auto start = std::chrono::high_resolution_clock::now();

  // Matrix multiply

  auto e4 = q.parallel_for(sycl::range<2>(N, N), [=](sycl::id<2> idx) {
    const auto i = idx[0];
    const auto j = idx[1];

    real_type sum = 0.;
    for (size_t k = 0; k < N; ++k) {
      sum += gpuA[i * N + k] * gpuB[k * N + j];
    }

    gpuC[i * N + j] = sum;
  });

  // Report benchmark

  q.wait();
  const auto elapsed = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Time: " << elapsed << std::endl;

  // Copy Output to Host

  q.memcpy(C.data(), gpuC, sizeof(real_type) * C.size()).wait();

  // Clean up & Results

  sycl::free(gpuA, q);
  sycl::free(gpuB, q);
  sycl::free(gpuC, q);

  return 0;
}
