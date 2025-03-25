# CPU

set(COMPILE_FLAGS "-fsycl -Wall")
set(LINK_FLAGS "-fsycl ")

# Nvidia Ampere 3050-3090 GPU

# set(COMPILE_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_86 -Wall")
# set(LINK_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_86")

file(GLOB TASKS_NUM_NAMES RELATIVE
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_CURRENT_SOURCE_DIR}/*
)

include_directories(${RNG_INCLUDE_DIRS})

message(STATUS "[#] EXECUTABLES / LIBS")

foreach(TASK_NUM_NAME IN LISTS TASKS_NUM_NAMES)
  set(TASK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${TASK_NUM_NAME})
  if(IS_DIRECTORY ${TASK_PATH})
    list(APPEND TASK_INCLUDE_DIRS ${TASK_PATH})
    string(REGEX REPLACE "^[0-9]+_" "" TASK_NAME ${TASK_NUM_NAME})
    if(EXISTS ${TASK_PATH}/${TASK_NUM_NAME}.cpp AND
       EXISTS ${TASK_PATH}/${TASK_NAME}.cpp AND
       EXISTS ${TASK_PATH}/${TASK_NAME}.h)
      add_executable(${TASK_NUM_NAME} ${TASK_PATH}/${TASK_NUM_NAME}.cpp)

      set_target_properties(${TASK_NUM_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
      set_target_properties(${TASK_NUM_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

      add_library(${TASK_NAME} ${TASK_PATH}/${TASK_NAME}.cpp 
                              ${TASK_PATH}/${TASK_NAME}.h)

      set_target_properties(${TASK_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
      set_target_properties(${TASK_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

      if(${TASK_NAME} MATCHES ".*mkl.*")
        target_compile_options(${TASK_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL_SYCL,INTERFACE_COMPILE_OPTIONS>)
        target_include_directories(${TASK_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL_SYCL,INTERFACE_INCLUDE_DIRECTORIES>)
        target_link_libraries(${TASK_NAME} PUBLIC $<LINK_ONLY:MKL::MKL_SYCL>) 

        target_compile_options(${TASK_NUM_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL_SYCL,INTERFACE_COMPILE_OPTIONS>)
        target_include_directories(${TASK_NUM_NAME} PUBLIC $<TARGET_PROPERTY:MKL::MKL_SYCL,INTERFACE_INCLUDE_DIRECTORIES>)
        target_link_libraries(${TASK_NUM_NAME} PUBLIC $<LINK_ONLY:MKL::MKL_SYCL>)
      endif()

      target_link_libraries(${TASK_NUM_NAME} PRIVATE 
        ${TASK_NAME}
        rng
      )

      message(STATUS "- ${TASK_NUM_NAME} / lib_${TASK_NAME}")
    endif()
  endif()
endforeach()

set(TASK_INCLUDE_DIRS ${TASK_INCLUDE_DIRS} PARENT_SCOPE)
