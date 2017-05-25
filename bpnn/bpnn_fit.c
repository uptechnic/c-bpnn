//
// Created by wcjzj on 2017/5/23.
//

#include "bpnn_fit.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#define T bpnn_t

#define D   IN_N
#define Q   HIDDEN_N
#define L   OUT_N

struct T {
    double *v[D];
    double *w[Q];
    double *r;
    double *o;
    double *b;
};

static bool get_parameter(T bpnn);

T bpnn_fit_new(void) {
    T bpnn;

    bpnn = malloc(sizeof(*bpnn));
    if (bpnn == NULL)
        goto cleanup;

    for (size_t i = 0; i < D; i++) {
        bpnn->v[i] = malloc(sizeof(double) * Q);
        if (bpnn->v[i] == NULL)
            goto cleanup;
    }

    for (size_t h = 0; h < Q; h++) {
        bpnn->w[h] = malloc(sizeof(double) * L);
        if (bpnn->w[h] == NULL)
            goto cleanup;
    }

    bpnn->r = malloc(sizeof(double) * Q);
    if (bpnn->r == NULL)
        goto cleanup;
    bpnn->o = malloc(sizeof(double) * L);
    if (bpnn->o == NULL)
        goto cleanup;
    bpnn->b = malloc(sizeof(double) * Q);
    if (bpnn->b == NULL)
        goto cleanup;

    if (!get_parameter(bpnn)) {
        printf("[BPNN] GET PARAMETER FAILED!\n");
        goto cleanup;
    }

    return bpnn;

    cleanup:
    if (bpnn) {
        for (size_t i = 0; i < D; i++) {
            if (bpnn->v[i] != NULL)
                free(bpnn->v[i]);
        }

        for (size_t h = 0; h < Q; h++) {
            if (bpnn->w[h] != NULL)
                free(bpnn->w[h]);
        }

        if (bpnn->r != NULL)
            free(bpnn->r);
        if (bpnn->o != NULL)
            free(bpnn->o);
        if (bpnn->b != NULL)
            free(bpnn->b);
        free(bpnn);
    }

    return NULL;
}

void bpnn_fit(T bpnn, double *in, double *out) {
    assert(bpnn && in && out);

    /* Compute b[h] */
    for (size_t h = 0; h < Q; h++) {
        double alpha_h = 0;
        for (size_t i = 0; i < D; i++)
            alpha_h += bpnn->v[i][h] * in[i];
        bpnn->b[h] = ACTIVATION_FUNC(alpha_h - bpnn->r[h]);
    }

    /* Compute out[j] */
    for (size_t j = 0; j < L; j++) {
        double beta_j = 0;
        for (size_t h = 0; h < Q; h++)
            beta_j += bpnn->w[h][j] * bpnn->b[h];
        out[j] = ACTIVATION_FUNC(beta_j - bpnn->o[j]);
    }
}

void bpnn_fit_free(T *bpnn) {
    assert(bpnn && *bpnn);

    for (size_t i = 0; i < D; i++)
        free((*bpnn)->v[i]);

    for (size_t h = 0; h < Q; h++)
        free((*bpnn)->w[h]);

    free((*bpnn)->r);
    free((*bpnn)->o);
    free((*bpnn)->b);

    *bpnn = NULL;
}

static bool get_parameter(T bpnn) {
#define BUFFER_SIZE     128
    FILE *in = NULL;
    in = fopen(SAVE_PARAM_PATH, "r");
    if (in == NULL) {
        fprintf(stderr, "[BPNN] OPEN FILE %s FAILED.\n", SAVE_PARAM_PATH);
        return false;
    }

    char buffer[BUFFER_SIZE];
    for (size_t i = 0; (i < 5) && (fgets(buffer, BUFFER_SIZE, in) != NULL); i++) {
        if (buffer[0] == '#')
            continue;
        else if (buffer[0] == 'D' && buffer[1] == '=') {
            if (strtol(buffer + 2, NULL, 0) != D) {
                goto cleanup;
            }
        } else if (buffer[0] == 'Q' && buffer[1] == '=') {
            if (strtol(buffer + 2, NULL, 0) != Q) {
                goto cleanup;
            }
        } else if (buffer[0] == 'L' && buffer[1] == '=') {
            if (strtol(buffer + 2, NULL, 0) != L) {
                goto cleanup;
            }
        }
    }

    for (size_t i = 0; i < D; i++)
        for (size_t h = 0; h < Q; h++) {
            if (fgets(buffer, BUFFER_SIZE, in) != NULL) {
                bpnn->v[i][h] = strtod(buffer, NULL);
            } else {
                goto cleanup;
            }
        }
    for (size_t h = 0; h < Q; h++)
        for (size_t j = 0; j < L; j++) {
            if (fgets(buffer, BUFFER_SIZE, in) != NULL) {
                bpnn->w[h][j] = strtod(buffer, NULL);
            } else {
                goto cleanup;
            }
        }
    for (size_t h = 0; h < Q; h++) {
        if (fgets(buffer, BUFFER_SIZE, in) != NULL) {
            bpnn->r[h] = strtod(buffer, NULL);
        } else {
            goto cleanup;
        }
    }
    for (size_t j = 0; j < L; j++) {
        if (fgets(buffer, BUFFER_SIZE, in) != NULL) {
            bpnn->o[j] = strtod(buffer, NULL);
        } else {
            goto cleanup;
        }
    }

    fclose(in);

    return true;

    cleanup:
    fprintf(stderr, "[BPNN] BPNN PARAM FILE NOT FIT!\n");
    fclose(in);
    return false;
}

#undef T
