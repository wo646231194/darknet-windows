#include "pyramid_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

pyramid_layer make_pyramid_layer(int batch, int inputs, int n, int level, int classes, int coords, int rescore)
{
    pyramid_layer l = {0};
    l.type = PYRAMID;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.level = level;
    //assert( pow(2,2*level) <= inputs/1024);
    l.truths = 0;
    for (int i = 0; i < level; i++){
        l.truths += pow(2, 2 * i);
    }
    l.truths = l.truths * (classes + coords);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.truths * n;
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));
#ifdef GPU
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Pyramid Layer: output %d , pyramid level %d , per area %d * (x,y,w,h,s) box\n", l.truths * n, l.level, l.n);
    srand(0);

    return l;
}

void forward_pyramid_layer(const pyramid_layer l, network_state state, int truth_index, int level)
{
    int i,j;
    //memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    int b;
    if(state.train){
        float avg_loc = 0;
        float avg_conf = 0;
        float avg_anyobj = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float conf_loss = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            int is_obj = state.truth[truth_index];

            if (!is_obj){
                for (j = 0; j < l.n; ++j) {
                    int p_index = index + j*(1 + l.coords);
                    float h_theta_x = 1 / (1 + exp(-state.input[p_index]));
                    l.delta[p_index] = l.noobject_scale*(0 - h_theta_x);
                    conf_loss -= log(1 - h_theta_x);
                }
                *l.cost += conf_loss;
                printf("neg %d ->  %f\n", level, *l.cost);
                continue;
            }

            int best_index = -1;
            float best_iou = 0;
            float best_rmse = INFINITY;

            for (j = 0; j < l.n; ++j) {
                int p_index = index + j*(1 + l.coords);
                float h_theta_x = 1 / (1 + exp(-state.input[p_index]));
                l.delta[p_index] = l.object_scale*(1 - h_theta_x);
                conf_loss -= log(h_theta_x);
            }

            box truth = float_to_box(state.truth + truth_index + 1 );

            for(j = 0; j < l.n; ++j){
                int box_index = index + j*(1 + l.coords)+1;
                box out = float_to_box(state.input + box_index);

                float rmse = box_smooth_loss_l1(out, truth);
                if(rmse < best_rmse){
                    best_rmse = rmse;
                    best_index = j;
                }
            }

            if(l.random && *(state.net.seen) < 64000){
                best_index = rand()%l.n;
            }

            int box_index = index + best_index * (1 + l.coords) + 1 ;
            box out = float_to_box(state.input + box_index);
            if (l.sqrt) {
                out.w = out.w*out.w;
                out.h = out.h*out.h;
            }
            float loss_l1 = box_smooth_loss_l1(out, truth);

            //printf("%d,", best_index);
            *(l.cost) += conf_loss + l.coord_scale * loss_l1;

            l.delta[box_index + 0] = l.coord_scale*(state.truth[truth_index + 0] - state.input[box_index + 0]);
            l.delta[box_index + 1] = l.coord_scale*(state.truth[truth_index + 1] - state.input[box_index + 1]);
            l.delta[box_index + 2] = l.coord_scale*(state.truth[truth_index + 2] - state.input[box_index + 2]);
            l.delta[box_index + 3] = l.coord_scale*(state.truth[truth_index + 3] - state.input[box_index + 3]);

            printf("pos %d ->  %f\n", level, *l.cost);
        }

        //*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);

        if (truth_index == 0){
            //printf("Pyramid loss -> 1 %f\n", *l.cost);
            //printf("Pyramid Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou / count, avg_cat / count, avg_allcat / (count*l.classes), avg_obj / count, avg_anyobj / (l.batch*l.n), count);
        }
    }
}

void backward_pyramid_layer(const pyramid_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_pyramid_layer_gpu(const pyramid_layer l, network_state state, int truth_index, int level)
{
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    if(!state.train){
        copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, (l.n* truth_index)+1);
        for (int i = 0; i < l.inputs; i++){
            l.output[truth_index *5 + i] = in_cpu[i];
        }
        return;
    }

    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.truths ;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_pyramid_layer(l, cpu_state, truth_index, level);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    //cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_pyramid_layer_gpu(pyramid_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif

