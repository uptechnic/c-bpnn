cmake_minimum_required(VERSION 3.7)
project(c-bpnn)

set(CMAKE_C_STANDARD 99)
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(bpnn)
include_directories(bpnn)

add_executable(test_bpnn_train test_bpnn_train.c)
target_link_libraries(test_bpnn_train BPNN m)
set_target_properties(test_bpnn_train PROPERTIES LINKER_LANGUAGE C)

add_executable(test_bpnn test_bpnn.c)
target_link_libraries(test_bpnn BPNN m)
set_target_properties(test_bpnn PROPERTIES LINKER_LANGUAGE C)