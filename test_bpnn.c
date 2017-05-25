//
// Created by wcjzj on 2017/5/23.
//

#include "bpnn_config.h"
#include "bpnn_fit.h"
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

static bool test_set_get(double *in, double *out);

static FILE *in_file = NULL;
static FILE *out_file = NULL;

int main(void) {

    bpnn_t bpnn = bpnn_fit_new();
    if (bpnn == NULL) {
        fprintf(stderr, "bpnn new failed!\n");
        return 0;
    }

    in_file = fopen(IN_PATH, "r");
    if (in_file == NULL) {
        fprintf(stderr, "open file %s failed.\n", IN_PATH);
        return 0;
    }

    out_file = fopen(OUT_PATH, "r");
    if (out_file == NULL) {
        fprintf(stderr, "open file %s failed.\n", OUT_PATH);
        return 0;
    }

    double in[IN_N], out[OUT_N], outy[OUT_N];
    while (test_set_get(in, out)) {
        bpnn_fit(bpnn, in, outy);
        for (size_t i = 0; i < IN_N; i++)
            printf("%lf ", in[i]);
        for (size_t j = 0; j < OUT_N; j++)
            printf("%lf[%lf] ", out[j], outy[j]);
        printf("\n");
    }

    bpnn_fit_free(&bpnn);

    fclose(in_file);
    fclose(out_file);

    return 0;
}

static bool test_set_get(double *in, double *out) {
    assert(in && out);
#define BUFFER_SIZE 128
    static char buffer[BUFFER_SIZE];

    if (in_file && out_file) {
        if (fgets(buffer, BUFFER_SIZE, in_file) != NULL) {
            char *token = strtok(buffer, ",");
            for (size_t i = 0; i < IN_N; i++) {
                if (token == NULL) {
                    fprintf(stderr, "the format of input is not correct!\n");
                    return false;
                }
                in[i] = strtod(token, NULL);
                token = strtok(NULL, ",");
            }

        } else {
            return false;
        }

        if (fgets(buffer, BUFFER_SIZE, out_file) != NULL) {
            out[0] = strtod(buffer, NULL);
        } else {
            return false;
        }

        return true;
    }

    return false;

}