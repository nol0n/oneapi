set(COMPILE_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_86 -Wall")
set(LINK_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_86")

add_executable(sample main.cpp)

set_target_properties(sample PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(sample PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
