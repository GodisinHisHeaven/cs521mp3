cmake_minimum_required(VERSION 3.14)
project(dual_ad_test)

set(CMAKE_CXX_STANDARD 17)

# Include FetchContent
include(FetchContent)

# Download and build GoogleTest automatically
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

# Your executable
add_executable(dual_test dual_test.cpp)

# Link to GoogleTest
target_link_libraries(dual_test gtest gtest_main pthread)