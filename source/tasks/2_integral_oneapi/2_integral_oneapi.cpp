#include <device.hpp>
#include <iostream>

#include "integral_oneapi.h"

int main() {
  sycl::device device = sycl::device(sycl::cpu_selector_v);
  std::cout << IntegralONEAPI(0, 1, 100, device) << "\n";

  return 0;
}