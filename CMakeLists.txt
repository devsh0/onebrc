cmake_minimum_required(VERSION 3.26)
project(onebrc)

set(CMAKE_CXX_STANDARD 17)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(onebrc main.cpp)
target_compile_options(onebrc PRIVATE -march=skylake)
target_link_libraries(onebrc PRIVATE Threads::Threads)
