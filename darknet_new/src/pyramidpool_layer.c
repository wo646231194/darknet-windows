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
    l.delta =   calloc(output_size, sizeof(float));
    #ifdef GPU
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
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

void forward_pyramidpool_layer(const pyramidpool_layer l,const convolutional_layer lc, network_state state, int now)
{
    int b,i,j,k,m,n;

    int h = lc.out_h;
    int w = lc.out_w;
    int c = lc.out_c;
    int level = sqrt(h)-1;
    state.index = now;
    layer connect = state.net.layers[now];
    layer dropout = state.net.layers[now + 1];
    layer pyramid = state.net.layers[now + 2];

    for(b = 0; b < lc.batch; ++b){
        for(i = 0; i < h; i+=l.size){
            for(j = 0; j < w; j+=l.size){
                for (k = 0; k < c; ++k){
                    int in_index = j + w*(i + h*(k + c*b));
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int out = n*l.size + m + k*(l.size*l.size);
                            int in = in_index + n*l.size + m;
                            l.output[out] = state.input[in];
                        }
                    }
                }
                cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
                state.input = l.output_gpu;
                forward_connected_layer_gpu(connect, state);
                state.input = connect.output_gpu;
                forward_dropout_layer_gpu(dropout, state);
                state.input = dropout.output_gpu;
                forward_pyramid_layer_gpu(pyramid, state);
            }
        }
    }
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
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.input = in_cpu;
    layer lc = state.net.layers[i - 1];
    for (int j = 0; j < l.level; j++){
        forward_pyramidpool_layer(l, lc, cpu_state, i+1);
        cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
        cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
        free(cpu_state.input);
    }
}

void backward_pyramidpool_layer_gpu(pyramidpool_layer l, network_state state){

}

#endif