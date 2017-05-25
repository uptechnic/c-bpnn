//
// Created by wcjzj on 2017/5/22.
//

#include <stdio.h>
#include <assert.h>
#include "bpnn.h"

static bool test_set_get(double *in, double *out);

static bool test_set_init(void);

static FILE *in_file = NULL;
static FILE *out_file = NULL;

int main(void) {
    in_file = fopen(TEST_IN_PATH, "r");
    if (in_file == NULL) {
        fprintf(stderr, "open file %s failed.\n", TEST_IN_PATH);
        return 0;
    }

    out_file = fopen(TEST_OUT_PATH, "r");
    if (out_file == NULL) {
        fprintf(stderr, "open file %s failed.\n", TEST_OUT_PATH);
        return 0;
    }

    bpnn_init();
    bpnn_train(test_set_get, test_set_init);

    fclose(in_file);
    fclose(out_file);

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

    bpnn_sim(test_set_get);

    fclose(in_file);
    fclose(out_file);

    return 0;
}

static bool test_set_get(double *in, double *out) {
    assert(in && out);
#define BUFFER_SIZE 256
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

static bool test_set_init(void) {
    fseek(in_file, 0, SEEK_SET);
    fseek(out_file, 0, SEEK_SET);
}
