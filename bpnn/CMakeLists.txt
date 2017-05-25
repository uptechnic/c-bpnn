cmake_minimum_required(VERSION 3.7)

set(BPNN_INC_FILES bpnn.h bpnn_config.h bpnn_fit.h)
set(BPNN_SRC_FILES bpnn.c bpnn_fit.c)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm")

add_library(BPNN ${BPNN_INC_FILES} ${BPNN_SRC_FILES})
set_target_properties(BPNN PROPERTIES LINKER_LANGUAGE C)
