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
    int k=0;
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
    //l.truths = 0;
    for (int i = 0; i < level; i++){
        k += pow(2, 2 * i);
    }
    l.truths = k * (classes + coords) * n;
    l.cost = calloc(1, sizeof(float));
    l.outputs = k * inputs;
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));
#ifdef GPU
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Pyramid Layer: output %d , pyramid level %d , per area %d * (x,y,w,h,s) box\n", k * inputs, l.level, l.n);
    srand(0);

    return l;
}

void forward_pyramid_layer(const pyramid_layer l, network_state state, int truth_index, int level)
{
    int i,j,t;
    //memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    int b, s, is_obj;
    if(state.train){
        int count = 0;
        float avg_loc = 0;
        float pos_loss = 0;
        float neg_loss = 0;
        float loc_loss = 0;
        *(l.cost) = 0;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (s = 0; s < l.n; ++s){
                t = truth_index*l.n + s * (1 + l.coords);
                is_obj = state.truth[t];

                if (!is_obj){
                    int p_index = index + s*(1 + l.coords + l.rescore);
                    float h_theta_x = state.input[p_index];
                    l.delta[p_index] = 0.0 - h_theta_x;
                    //float h_theta_x = 1 / (1 + exp(-state.input[p_index]));
                    //l.delta[p_index] = l.noobject_scale*(0 - state.input[p_index]);
                    //l.delta[p_index] = l.noobject_scale*(0 - h_theta_x);
                    //conf_loss += pow(l.delta[p_index] + 1, 2);
                    neg_loss -= log(1 - h_theta_x);
                    //l.delta[p_index + 1] = 0-state.input[p_index + 1];
                    //l.delta[p_index + 2] = 0-state.input[p_index + 2];
                    //l.delta[p_index + 3] = 0-state.input[p_index + 3];
                    //l.delta[p_index + 4] = 0-state.input[p_index + 4];
                    //loc_loss = loc_loss + smooth_l1(state.input[p_index + 1], l.coord_scale) +
                    //    smooth_l1(state.input[p_index + 2], l.coord_scale) +
                    //    smooth_l1(state.input[p_index + 3], l.coord_scale) +
                    //    smooth_l1(state.input[p_index + 4], l.coord_scale);
                    //*l.cost = conf_loss ;
                    //printf("neg %d  ->  all  %.5f, conf  %.5f, loc %.5f  \n", level, *l.cost, conf_loss, loc_loss / 4);
                    continue;
                }

                int best_index = -1;
                float best_iou = 0;
                float best_rmse = INFINITY;

                //for (j = 0; j < l.n; ++j) {
                //    int p_index = index + j*(1 + l.coords);
                //    float h_theta_x = 1 / (1 + exp(-state.input[p_index]));
                //    l.delta[p_index] = l.object_scale*(1 - h_theta_x);
                //    conf_loss -= log(h_theta_x);
                //}

                box truth = float_to_box(state.truth + t + 1);

                //for (j = 0; j < l.n; ++j){
                //    int box_index = index + j*(1 + l.coords) + 1;
                //    box out = float_to_box(state.input + box_index);
                //    out.h = out.h * out.h;
                //    out.w = out.h / 3.2;

                //    float iou = box_iou(out, truth);
                //    float rmse = box_rmse(out, truth);
                //    if (best_iou>0 || iou > 0){
                //        if (iou > best_iou){
                //            best_iou = iou;
                //            best_index = j;
                //        }
                //    }
                //    else{
                //        if (rmse < best_rmse){
                //            best_rmse = rmse;
                //            best_index = j;
                //        }
                //    }
                //}

                //if(l.random && *(state.net.seen) < 64000){
                //    best_index = rand()%l.n;
                //}
                best_index = s;
                //printf("%d , ", s);

                int box_index = index + best_index * (1 + l.coords + l.rescore) + 1 + l.rescore;
                l.delta[box_index - 2] = 1.0 - state.input[box_index - 2];
                box out = float_to_box(state.input + box_index);
                float step = 1.0 / l.n;

                out.x = out.x / l.n;
                out.x = out.x + step * s;
                out.h = out.h / 2.0;
                out.h = out.h + 0.5;
                out.w = out.h / 3.2;

                //loc_loss = box_smooth_loss_l1(out, truth, l.coord_scale);
                float iou = box_iou(out, truth);
                loc_loss += 1. - iou;

                float h_theta_x = state.input[box_index-1];
                //l.delta[box_index - 1] = l.object_scale*(iou - state.input[box_index - 1]);
                l.delta[box_index -1] = l.object_scale*(iou - h_theta_x);
                pos_loss = pos_loss - log(h_theta_x) - log(state.input[box_index - 2]);
                //conf_loss += pow(l.delta[box_index - 1] + loc_loss, 2);

                //printf("%d,", best_index);
                //*(l.cost) += conf_loss + loc_loss;
                //for (int t = 0; t < 4; t++){
                //l.delta[box_index + t] = smooth_l1(state.truth[truth_index + t] - state.input[box_index + t], l.coord_scale);
                //l.delta[box_index + t] = l.coord_scale*(state.truth[truth_index + 0] - state.input[box_index + 0]);
                //}
                //truth.x = truth.x - step * s;
                l.delta[box_index + 0] = l.coord_scale*(truth.x - out.x)*l.n;
                l.delta[box_index + 1] = l.coord_scale*(truth.y - out.y);
                //l.delta[box_index + 2] = l.coord_scale*(sqrt(truth.w) - state.input[box_index + 2]);
                l.delta[box_index + 3] = l.coord_scale*(truth.h - out.h)*2.0;

                count++;
            }
            pos_loss = pos_loss / count;
            loc_loss = loc_loss / count;
            neg_loss = neg_loss / (l.n - count);
            *(l.cost) += pos_loss + loc_loss + neg_loss;
            if (loc_loss){
                printf("p   %d  ->  all  %.5f, pos  %.5f, loc %.5f, neg %.5f  \n", level, *l.cost, pos_loss, loc_loss, neg_loss);
            }
            else{
                printf("neg %d  ->  all  %.5f, pos  %.5f, loc %.5f, neg %.5f  \n", level, *l.cost, pos_loss, loc_loss, neg_loss);
            }
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

    float *truth_cpu = 0;
    if (state.truth){
        int num_truth = l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }

    if(! state.train){
        truth_index = (int)truth_index * 1.2;
        copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, (l.n* truth_index)+1);
        //in_cpu[0] = 1; in_cpu[1] = 0.5; in_cpu[2] = 0.5; in_cpu[3] = 0.5; in_cpu[4] = 0; in_cpu[5] = 1;
        //in_cpu[6] = 1; in_cpu[7] = 0.5; in_cpu[8] = 0.5; in_cpu[9] = 0.5; in_cpu[10] = 0; in_cpu[11] = 1;
        for (int i = 0; i < l.inputs; i++){
            l.output[truth_index *l.n + i] = in_cpu[i];
        }
        //copy_ongpu(l.batch*5, state.truth, truth_index , l.output_gpu, (l.n* truth_index) );
        return;
    }

    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_pyramid_layer(l, cpu_state, truth_index, level);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_pyramid_layer_gpu(pyramid_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif

