#include "pyramidpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_pyramidpool_image(pyramidpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_pyramidpool_delta(pyramidpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

pyramidpool_layer make_pyramidpool_layer(int batch, int h, int w, int c, int level, int size)
{
    //fprintf(stderr, "Maxpool Layer: %d x %d x %d image, %d size, %d stride\n", h,w,c,size,stride);
    pyramidpool_layer l = {0};
    l.type = PYRAMIDPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = size;
    l.out_h = size;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.level = level;
    l.size = size;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta = calloc(l.inputs, sizeof(float));
    #ifdef GPU
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, l.inputs);
    #endif
    fprintf(stderr, "Pyramidpool Layer: %d level, %d size -> %d x %d x %d image\n", level, size, l.out_h, l.out_w, l.out_c);
    return l;
}

void resize_pyramidpool_layer(pyramidpool_layer *l, int w, int h)
{
    int stride = l->stride;
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w-1)/stride + 1;
    l->out_h = (h-1)/stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

int get_truth_index(int level, int size, int x, int y)
{
    int i,index=0;
    for (i = 0; i < level; i++){
        index += pow(2, 2 * i);
    }
    index += x / size *pow(2, level) + y / size;
    return index * 5;
}

void forward_pyramidpool_layer(float * incpu, layer l, pyramidpool_layer py, network_state state, int now, float *delta)
{    
    int b,i,j,k,m,n;
    int h, w, c, th, level=0, in, out;

    if (l.type == PYRAMIDPOOL){
        h = l.h;
        w = l.w;
        c = l.c;
    } else{
        h = l.out_h;
        w = l.out_w;
        c = l.out_c;
    }

    th = h/2;
    while (th>1){
        th /= 2;
        level++;
    }
    
    state.index = now;

    for(b = 0; b < l.batch; ++b){
        for(i = 0; i < h; i+=py.size){
            for(j = 0; j < w; j+=py.size){
                for (k = 0; k < c; ++k){
                    int in_index = j + w*(i + h*(k + c*b));
                    for(n = 0; n < py.size; ++n){
                        for(m = 0; m < py.size; ++m){
                            out = n*py.size + m + k*(py.size*py.size);
                            in = in_index + n*l.w + m;
                            l.output[out] = incpu[in];
                        }
                    }
                }
                int truth_index = get_truth_index(level, l.size, i, j);
                cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
                state.input = l.output_gpu;
                forward_network_pyramid_gpu(state.net, state, now , truth_index);
                backward_network_pyramid_gpu(state.net, state, now);
                if (l.level > 0){
                    cuda_pull_array(py.delta_gpu, py.delta, py.batch*py.outputs);
                    for (k = 0; k < c; ++k){
                        int in_index = j + w*(i + h*(k + c*b));
                        for (n = 0; n < py.size; ++n){
                            for (m = 0; m < py.size; ++m){
                                out = n*py.size + m + k*(py.size*py.size);
                                in = in_index + n*l.w + m;
                                //delta[in] = py.delta[out];
                            }
                        }
                    }
                }
                /*else{
                    for (k = 0; k < c; ++k){
                        int in_index = j + w*(i + h*(k + c*b));
                        for (n = 0; n < py.size; ++n){
                            for (m = 0; m < py.size; ++m){
                                out = n*py.size + m + k*(py.size*py.size);
                                int stride = pow(2, (py.level - level - 1));
                                in = (in_index + n*py.w + m) * stride;
                                for (int t = 0; t < stride; t++){
                                    delta[in++] += py.delta[out]/stride;
                                }
                            }
                        }
                    }
                }*/
            }
        }
    }
    int ttt = 0;
}

void backward_pyramidpool_layer(const pyramidpool_layer l, network_state state)
{
    int i;
    int h = (l.h-1)/l.stride + 1;
    int w = (l.w-1)/l.stride + 1;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}

#ifdef GPU

void forward_pyramidpool_layer_gpu(pyramidpool_layer l, network_state state, int i)
{
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *delta_cpu = calloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.input = in_cpu;
    layer layer = l;
    for (int j = 1; j < l.level; j++){
        forward_pyramidpool_layer(in_cpu, layer, l, cpu_state, i+1, delta_cpu);
        state.input = layer.output_gpu;
        if (layer.type == PYRAMIDPOOL){
            layer = make_maxpool_layer_show(l.batch, layer.h, layer.w, layer.c, l.size, l.size, 0);
        } else{
            layer = make_maxpool_layer_show(l.batch, layer.out_h, layer.out_w, layer.out_c, l.size, l.size, 0);
        }
        forward_maxpool_layer_gpu(layer, state);
        state.input = layer.output_gpu;
        cuda_pull_array(state.input, in_cpu, layer.batch*layer.outputs);
    }
    free(cpu_state.input);
    cuda_push_array(l.delta_gpu, delta_cpu, l.batch*l.inputs);
    free(delta_cpu);
}

void backward_pyramidpool_layer_gpu(pyramidpool_layer l, network_state state){
    int i;
    constrain_ongpu(l.outputs*l.batch, 5, l.delta_gpu, 1);
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    for (i = 0; i < l.batch; ++i){
        axpy_ongpu(l.outputs, 1, l.delta_gpu + i*l.outputs, 1, l.bias_updates_gpu, 1);
    }

    if (l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
    }

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = state.input;
    float * c = l.weight_updates_gpu;
    gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = state.delta;

    if (c) gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}

#endif