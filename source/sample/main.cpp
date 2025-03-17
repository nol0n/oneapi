#include <device_selector.hpp>
#include <exception.hpp>
#include <exception>
#include <exception_list.hpp>
#include <vector>

#include <assert.h>

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 1024;

int main(int argc, char *argv[]) {

  std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);
  try {
    buffer<float> buf_a(a.data(), a.size());
    buffer<float> buf_b(b.data(), b.size());
    buffer<float> buf_c(c.data(), c.size());

    auto selector = gpu_selector_v;

    queue queue(selector, [](sycl::exception_list list) {
      for (auto &e : list) {
        try {
          std::rethrow_exception(e);
        } catch (sycl::exception &e) {
          std::cout << "Asynchronous error: " << e.what() << std::endl;
        }
      }
    });

    queue.submit([&](handler &cgh) {
      stream s(1024, 80, cgh);
      auto in_a = buf_a.get_access<access::mode::read>(cgh);
      auto in_b = buf_b.get_access<access::mode::read>(cgh);
      auto out_c = buf_c.get_access<access::mode::write>(cgh);

      cgh.parallel_for(range<1>(N), [=](id<1> i) {
        out_c[i] = in_a[i] + in_b[i];
        if (i == 0) {
          s << "Hello" << sycl::endl;
        }
      });
    });

    queue.wait_and_throw();
  } catch (sycl::exception e) {
    std::cout << "Synchronous error: " << e.what() << std::endl;
  } catch (...) {
    std::cout << "General error" << std::endl;
  }

  for (int i = 0; i < N; ++i) {
    assert(c[i] == 3.0f);
  }
  return 0;
}
