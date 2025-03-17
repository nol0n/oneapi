#include <chrono>
#include <device.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

#include "jacobi_shared_oneapi.h"
#include "rng.hpp"

int main() {
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  std::chrono::duration<double> seconds;
  std::cout << std::setprecision(3);

  size_t n = 2000;
  std::vector<float> a = rng::diag_dominant(n, 0.0f, 10.0f);
  std::vector<float> b = rng::float_vector(n, 100.0f, 300.0f);

  size_t n_progrev = 2;
  std::vector<float> a_progrev = rng::diag_dominant(n_progrev, 0.0f, 10.0f);
  std::vector<float> b_progrev = rng::float_vector(n_progrev, 100.0f, 300.0f);

  sycl::device device = sycl::device(sycl::cpu_selector_v);

  //==================================================================

  std::vector<float> mine = JacobiSharedONEAPI(a_progrev, b_progrev, 0.001f, device);
  
  // time
  start = std::chrono::high_resolution_clock::now();
  mine = JacobiSharedONEAPI(a, b, 0.001f, device);
  end = std::chrono::high_resolution_clock::now();

  seconds = end - start;
  std::cout << " - " << seconds.count() << " s.\n";

  for (int i = 50; i < 55; i++)
    std::cout << mine[i] << " ";
  std::cout << "\n";

  //==================================================================

  return 0;
}