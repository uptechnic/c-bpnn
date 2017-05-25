//
// Created by wcjzj on 2017/5/22.
//

#ifndef FETALHEART_BPNN_H
#define FETALHEART_BPNN_H

/**
 * Module for implements Back Propagation Neural Networks(BPNN).
 * Including init, train, sim functions.
 */

#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "bpnn_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * To fetch test_set.
 * The length of in is IN_N, the length of out is OUT_N.
 */
typedef bool (*test_set_get_t)(double *in, double *out);

/**
 * Reset test_set fetch process.
 */
typedef bool (*test_set_init_t)(void);

/**
 * Init bpnn module.
 */
void bpnn_init(void);

/**
 * Train bpnn module and produce parameter file.
 * @param f_get To get test data in stream.
 */
void bpnn_train(test_set_get_t f_get, test_set_init_t f_init);

/**
 * Test result of bpnn train parameter.
 * @param f_get
 */
void bpnn_sim(test_set_get_t f_get);

#ifdef __cplusplus
}
#endif

#endif //FETALHEART_BPNN_H
