if(LITE_ON_MODEL_OPTIMIZE_TOOL)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  set(lite_kernel_deps ${lite_kernel_deps} math_host CACHE INTERNAL "")
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
endif()

message(STATUS "the global varible KERNELS_SRC is ${KERNELS_SRC}")
set(KERNELS_SRC "")
message(STATUS "compile with lite host kernels")


