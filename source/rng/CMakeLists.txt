add_library(rng STATIC rng.cpp)

target_link_libraries(rng PRIVATE OpenMP::OpenMP_CXX)

set(RNG_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
