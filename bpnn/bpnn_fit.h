//
// Created by wcjzj on 2017/5/23.
//

#ifndef FETALHEART_BPNN_FIT_H
#define FETALHEART_BPNN_FIT_H

#include "bpnn_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define T bpnn_t
typedef struct T *T;

/**
 * Init bpnn module.
 * @return
 */
T bpnn_fit_new(void);

/**
 * Using bpnn fit in to out.
 * @param bpnn
 * @param in
 * @param out
 */
void bpnn_fit(T bpnn, double *in, double *out);

/**
 * Uninit bpnn.
 * @param bpnn
 */
void bpnn_fit_free(T *bpnn);

#undef T

#ifdef __cplusplus
}
#endif

#endif //FETALHEART_BPNN_FIT_H
