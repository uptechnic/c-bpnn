//
// Created by wcjzj on 2017/5/22.
//

#include "bpnn.h"
#include <time.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define RAND                    (rand() / (double)(RAND_MAX))

#define D   IN_N
#define Q   HIDDEN_N
#define L   OUT_N
#define U1  LEARN_RATE1
#define U2  LEARN_RATE2

static double v[D][Q];
static double w[Q][L];
static double r[Q];
static double o[L];

static double g[L];
static double e[Q];

static double x[D];
static double b[Q];
static double y[L];
static double yc[L];

static double Ek;

static bool train_once(test_set_get_t f_get);

static bool save_parameter(void);

static bool read_parameter(void);

void bpnn_init(void) {
    srand((unsigned) time(NULL));
    for (size_t i = 0; i < D; i++)
        for (size_t h = 0; h < Q; h++)
            v[i][h] = RAND;
    for (size_t h = 0; h < Q; h++)
        for (size_t j = 0; j < L; j++)
            w[h][j] = RAND;
    for (size_t h = 0; h < Q; h++)
        r[h] = RAND;
    for (size_t j = 0; j < L; j++)
        o[j] = RAND;
}

void bpnn_train(test_set_get_t f_get, test_set_init_t f_init) {
    assert(f_get && f_init);

    printf("[BPNN] TRAIN STARTING...\n");

    f_init();

    int train_c = 0;
    double E = 0;
    while (train_c < LOOP_N) {
        size_t count = 0;
        while (train_once(f_get)) {
            count++;
            E += Ek;
        }
        E /= count;
        train_c++;
        printf("[BPNN] LOOP %d E %lf\n", train_c, E);
        f_init();
        if (E < E_MIN)
            break;
    }

    printf("[BPNN] END TRAIN PROCESS LOOP %d E %lf .\n", train_c, E);

    if (save_parameter())
        printf("[BPNN] SAVE BPNN PARAM SUCCESS in %s\n", SAVE_PARAM_PATH);
    else
        printf("[BPNN] SAVE BPNN PARAM FAILED.\n");
}

void bpnn_sim(test_set_get_t f_get) {
    assert(f_get);

    double maxEk = 0, minEk = 1, E = 0;
    int count = 0;
    while (f_get(x, y)) {
        count++;

        /* Compute b[h] */
        for (size_t h = 0; h < Q; h++) {
            double alpha_h = 0;
            for (size_t i = 0; i < D; i++)
                alpha_h += v[i][h] * x[i];
            b[h] = ACTIVATION_FUNC(alpha_h - r[h]);
        }

        double Ek = 0;

        /* Compute yc[j] */
        for (size_t j = 0; j < L; j++) {
            double beta_j = 0;
            for (size_t h = 0; h < Q; h++)
                beta_j += w[h][j] * b[h];
            yc[j] = ACTIVATION_FUNC(beta_j - o[j]);
            Ek += (yc[j] - y[j]) * (yc[j] - y[j]);
        }

        Ek = 0.5 * Ek;
        printf("[BPNN] DATA INDEX %d EK %lf ==> ", count, Ek);
        printf("INPUT: ");
        for (size_t i = 0; i < D; i++)
            printf("%lf ", x[i]);
        printf("OUTPUT:");
        for (size_t j = 0; j < L; j++)
            printf("%lf[%lf] ", y[j], yc[j]);
        printf("\n");
        maxEk = (Ek > maxEk) ? Ek : maxEk;
        minEk = (Ek < minEk) ? Ek : minEk;
        E += Ek;
    }

    E = E / count;

    printf("maxEk %lf, minEk %lf, E %lf\n", maxEk, minEk, E);
}

static bool train_once(test_set_get_t f_get) {
    if (!f_get(x, y))
        return false;

    /* Compute b[h] */
    for (size_t h = 0; h < Q; h++) {
        double alpha_h = 0;
        for (size_t i = 0; i < D; i++)
            alpha_h += v[i][h] * x[i];
        b[h] = ACTIVATION_FUNC(alpha_h - r[h]);
    }

    /* Compute yc[j] */
    for (size_t j = 0; j < L; j++) {
        double beta_j = 0;
        for (size_t h = 0; h < Q; h++)
            beta_j += w[h][j] * b[h];
        yc[j] = ACTIVATION_FUNC(beta_j - o[j]);
    }

    Ek = 0;

    /* Compute g[j] and E */
    for (size_t j = 0; j < L; j++) {
        g[j] = yc[j] * (1 - yc[j]) * (y[j] - yc[j]);
        Ek += (yc[j] - y[j]) * (yc[j] - y[j]);
    }

    Ek = 0.5 * Ek;

    /* Compute e[h] */
    for (size_t h = 0; h < Q; h++) {
        double temp = 0;
        for (size_t j = 0; j < L; j++)
            temp += w[h][j] * g[j];
        e[h] = b[h] * (1 - b[h]) * temp;
    }

    /* Update v[i][h], w[h][j], r[h], o[j] */
    for (size_t i = 0; i < D; i++)
        for (size_t h = 0; h < Q; h++)
            v[i][h] += U2 * e[h] * x[i];
    for (size_t h = 0; h < Q; h++)
        for (size_t j = 0; j < L; j++)
            w[h][j] += U1 * g[j] * b[h];
    for (size_t h = 0; h < Q; h++)
        r[h] += ((-U2) * e[h]);
    for (size_t j = 0; j < L; j++)
        o[j] += ((-U1) * g[j]);

    return true;
}

static bool save_parameter(void) {
    FILE *out = NULL;
    out = fopen(SAVE_PARAM_PATH, "w+");
    if (out == NULL) {
        fprintf(stderr, "[BPNN] OPEN FILE %s FAILED.\n", SAVE_PARAM_PATH);
        return false;
    }

    fprintf(out, "#BPNN PARAM GENERATED AUTOMATICALLY IN %d SECONDS SINCE JABUARY 1, 1970\n", (int)time(NULL));
    fprintf(out, "#!!DO NOT MODIFY!!\n");
    fprintf(out, "D=%d\n", D);
    fprintf(out, "Q=%d\n", Q);
    fprintf(out, "L=%d\n", L);
    for (size_t i = 0; i < D; i++)
        for (size_t h = 0; h < Q; h++)
            fprintf(out, "%lf\n", v[i][h]);
    for (size_t h = 0; h < Q; h++)
        for (size_t j = 0; j < L; j++)
            fprintf(out, "%lf\n", w[h][j]);
    for (size_t h = 0; h < Q; h++)
        fprintf(out, "%lf\n", r[h]);
    for (size_t j = 0; j < L; j++)
        fprintf(out, "%lf\n", o[j]);

    fclose(out);

    return true;
}