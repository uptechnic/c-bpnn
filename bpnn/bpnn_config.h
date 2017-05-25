//
// Created by wcjzj on 2017/5/22.
//

#ifndef FETALHEART_BPNN_CONFIG_H
#define FETALHEART_BPNN_CONFIG_H

#define IN_N                            3           /* INPUT NODE */
#define OUT_N                           1           /* OUTPUT_NODE */
#define HIDDEN_N                        71          /* HIDDEN_NODE */
#define LOOP_N                          5000        /* LOOP NUMBER */
#define E_MIN                           0.000001    /* Cumulative error */
#define LEARN_RATE1                     0.3
#define LEARN_RATE2                     0.4

#define ACTIVATION_FUNC(x)              (1/(1+exp(-(x))))

#define TEST_IN_PATH                    "../dataset/test_in.txt"
#define TEST_OUT_PATH                   "../dataset/test_out.txt"
#define IN_PATH                         "../dataset/in.txt"
#define OUT_PATH                        "../dataset/out.txt"
#define SAVE_PARAM_PATH                 "../dataset/bpnn_param.txt"

#endif //FETALHEART_BPNN_CONFIG_H
