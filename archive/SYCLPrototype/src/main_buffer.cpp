#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace hipsycl;

int main(int argc, char **argv) {
  constexpr size_t N = 3;

  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> B = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<double> C(N * N, 0.);

  sycl::queue q(sycl::gpu_selector{});

  {
    sycl::buffer<double, 2> bufferA(A.data(), sycl::range<2>(N, N));
    sycl::buffer<double, 2> bufferB(B.data(), sycl::range<2>(N, N));
    sycl::buffer<double, 2> bufferC(C.data(), sycl::range<2>(N, N));

    q.submit([&](sycl::handler &h) {
       auto a = bufferA.get_access<sycl::access::mode::read>(h);
       auto b = bufferB.get_access<sycl::access::mode::read>(h);
       auto c = bufferC.get_access<sycl::access::mode::write>(h);

       h.parallel_for(sycl::range<2>(N, N), [=](sycl::id<2> idx) {
         const auto i = idx[0];
         const auto j = idx[1];

         double sum = 0.;
         for (size_t k = 0; k < N; ++k) {
           sum += a[i][k] * b[k][j];
         }

         c[i][j] = sum;
       });
     }).wait();
  }

  for (const auto i : C)
    std::cout << i << ' ';
  std::cout << std::endl;

  return 0;
}
