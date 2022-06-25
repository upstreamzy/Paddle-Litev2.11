
set(CMAKE_SYSTEM_NAME Linux)



# Toolchain
if(RISCV64_TARGET_OS STREQUAL "riscv64linux")
    set(CMAKE_SYSTEM_PROCESSOR riscv64)
    set(CMAKE_C_COMPILER "riscv64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER "riscv64-linux-gnu-g++")
endif()

set(HOST_C_COMPILER $ENV{CC})
set(HOST_CXX_COMPILER $ENV{CXX})
if(NOT ${HOST_C_COMPILER})
    message(STATUS "HOST_C_COMPILER------------------------------------------------------------: ${HOST_C_COMPILER}")
    set(CMAKE_C_COMPILER ${HOST_C_COMPILER})
endif()
if(NOT ${HOST_CXX_COMPILER})
    message(STATUS "HOST_CXX_COMPILER------------------------------------------------------------: ${HOST_CXX_COMPILER}")
    set(CMAKE_CXX_COMPILER ${HOST_CXX_COMPILER})
endif()
message(STATUS "riscv64linux CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "riscv64linux CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# Definitions
add_definitions(-DLITE_WITH_LINUX)