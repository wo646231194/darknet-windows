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

void forward_pyramid_layer(const pyramid_layer l, network_state state, int truth_index)
{
    int i,j;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    int b;
    if(state.train){
        float avg_loc = 0;
        float avg_cat = 0;
        float avg_allcat = 0;
        float avg_conf = 0;
        float avg_anyobj = 0;
        int count = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            int is_obj = state.truth[truth_index];

            for (j = 0; j < l.n; ++j) {
                int p_index = index + j*(l.rescore + l.coords);
                l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                *(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                avg_anyobj += l.output[p_index];
            }

            int best_index = -1;
            float best_iou = 0;
            float best_rmse = INFINITY;

            if (!is_obj){
                continue;
            }

            for (j = 0; j < l.n; ++j) {
                int p_index = index + j*(l.rescore + l.coords);
                l.delta[p_index] = l.object_scale*(1 - l.output[p_index]);
                *(l.cost) += l.object_scale*pow(l.output[p_index], 2);
                avg_conf += l.output[p_index];
            }

            box truth = float_to_box(state.truth + truth_index + 1 );

            for(j = 0; j < l.n; ++j){
                int box_index = index + j*(l.rescore + l.coords)+1;
                box out = float_to_box(l.output + box_index);

                float rmse = box_loss_l1(out, truth);
                if(rmse < best_rmse){
                    best_rmse = rmse;
                    best_index = j;
                }
            }

            if(l.random && *(state.net.seen) < 64000){
                best_index = rand()%l.n;
            }

            int box_index = index + best_index * (l.rescore + l.coords) + 1 ;
            box out = float_to_box(l.output + box_index);
            if (l.sqrt) {
                out.w = out.w*out.w;
                out.h = out.h*out.h;
            }
            float loss_l1  = box_loss_l1(out, truth);

            //printf("%d,", best_index);
            *(l.cost) -= l.noobject_scale * pow(l.output[box_index], 2);
            *(l.cost) += l.object_scale * pow(1 - l.output[box_index], 2);
            avg_conf += l.output[box_index];
            l.delta[box_index] = l.object_scale * (1. - l.output[box_index]);

            if(l.rescore){
                //l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
            }

            l.delta[box_index + 0] = l.coord_scale*(state.truth[truth_index + 0] - l.output[box_index + 0]);
            l.delta[box_index + 1] = l.coord_scale*(state.truth[truth_index + 1] - l.output[box_index + 1]);
            l.delta[box_index + 2] = l.coord_scale*(state.truth[truth_index + 2] - l.output[box_index + 2]);
            l.delta[box_index + 3] = l.coord_scale*(state.truth[truth_index + 3] - l.output[box_index + 3]);
            if(l.sqrt){
                l.delta[box_index + 2] = l.coord_scale*(sqrt(state.truth[truth_index + 2]) - l.output[box_index + 2]);
                l.delta[box_index + 3] = l.coord_scale*(sqrt(state.truth[truth_index + 3]) - l.output[box_index + 3]);
            }

            *(l.cost) += loss_l1;
            avg_conf += loss_l1;
            ++count;
        }

        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);

        if (truth_index == 0){
            //printf("Pyramid Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou / count, avg_cat / count, avg_allcat / (count*l.classes), avg_obj / count, avg_anyobj / (l.batch*l.n), count);
        }
    }
}

void backward_pyramid_layer(const pyramid_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_pyramid_layer_gpu(const pyramid_layer l, network_state state, int truth_index)
{
    if(!state.train){
        copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
        return;
    }

    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.truths ;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_pyramid_layer(l, cpu_state, truth_index);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_pyramid_layer_gpu(pyramid_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif

