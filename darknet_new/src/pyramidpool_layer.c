#include "pyramidpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_pyramidpool_image(pyramidpool_layer l)
{
    int h = l.layer.out_h;
    int w = l.layer.out_w;
    int c = l.layer.c;
    return float_to_image(w,h,c,l.layer.output);
}

image get_pyramidpool_delta(pyramidpool_layer l)
{
    int h = l.layer.out_h;
    int w = l.layer.out_w;
    int c = l.layer.c;
    return float_to_image(w, h, c, l.layer.delta);
}

pyramidpool_layer make_pyramidpool_layer(int batch, int h, int w, int c, int level, int size, int pad)
{
    //fprintf(stderr, "Maxpool Layer: %d x %d x %d image, %d size, %d stride\n", h,w,c,size,stride);
    pyramidpool_layer l = {0};
    l.type = PYRAMIDPOOL;
    l.layer.type = PYRAMIDPOOL;
    l.layer.batch = batch;
    l.layer.h = h;
    l.layer.w = w;
    l.layer.c = c;
    l.layer.out_w = size+2*pad;
    l.layer.out_h = size+2*pad;
    l.layer.out_c = c;
    l.layer.outputs = l.layer.out_h * l.layer.out_w * l.layer.out_c;
    l.layer.inputs = h*w*c;
    l.layer.level = level;
    l.layer.size = size;
    l.layer.pad = pad;
    int output_size = l.layer.out_h * l.layer.out_w * l.layer.out_c * batch;
    l.layer.indexes = calloc(output_size, sizeof(int));
    l.layer.output = calloc(output_size, sizeof(float));
    l.layer.delta = calloc(l.layer.inputs, sizeof(float));
    #ifdef GPU
    l.layer.indexes_gpu = cuda_make_int_array(output_size);
    l.layer.output_gpu = cuda_make_array(l.layer.output, output_size);
    l.layer.delta_gpu = cuda_make_array(l.layer.delta, l.layer.inputs);
    #endif
    fprintf(stderr, "Pyramidpool Layer: %d level, %d size -> %d x %d x %d image\n", level, size, l.layer.out_h, l.layer.out_w, l.layer.out_c);
    for (int i = 1; i < l.layer.level; i++){
        l.maxpool[i - 1] = make_maxpool_layer_show(l.layer.batch, h, w, c, l.layer.size, l.layer.size, 1);
        h /= 2; w /= 2;
    }
    return l;
}

void resize_pyramidpool_layer(pyramidpool_layer *l, int w, int h)
{
    int stride = l->layer.stride;
    l->layer.h = h;
    l->layer.w = w;
    l->layer.inputs = h*w*l->layer.c;

    l->layer.out_w = (w - 1) / stride + 1;
    l->layer.out_h = (h - 1) / stride + 1;
    l->layer.out_h = (h - 1) / stride + 1;
    l->layer.outputs = l->layer.out_w * l->layer.out_h * l->layer.c;
    int output_size = l->layer.outputs * l->layer.batch;

    l->layer.indexes = realloc(l->layer.indexes, output_size * sizeof(int));
    l->layer.output = realloc(l->layer.output, output_size * sizeof(float));
    l->layer.delta = realloc(l->layer.delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->layer.indexes_gpu);
    cuda_free(l->layer.output_gpu);
    cuda_free(l->layer.delta_gpu);
    l->layer.indexes_gpu = cuda_make_int_array(output_size);
    l->layer.output_gpu = cuda_make_array(l->layer.output, output_size);
    l->layer.delta_gpu = cuda_make_array(l->layer.delta, output_size);
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

int truth_to_level(int n, int index, int *x, int *y){
    int sum = 0;
    for (int i = 0; i < n; i++){
        sum += pow(2, 2 * i);
        if (index < sum * 5) {
            sum -= pow(2, 2 * i);
            int in = index / 5 - sum;
            int col = pow(2, i);
            *x = in / col;
            *y = in % col;
            return i;
        }
    }
}

void get_truth_xy(float *truth, int *num_truth, int *truth_x, int *truth_y, int *level, int n){
    int num = 0;
    for (int i = 0; i < *num_truth; i+=5){
        if (truth[i]){
            level[num] = truth_to_level(n, floor(i/4), &truth_x[num], &truth_y[num]);
            num++;
        }
    }
    *num_truth = num;
}

void forward_pyramidpool_layer(float * incpu, layer l, layer py, network_state state, int now, float *delta, int x, int y)
{
    int b,i,j,k,m,n;
    int h, w, c, th, level=0, in, out;
    int pad = py.pad;

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
    srand((unsigned int)time(0));
    for(b = 0; b < l.batch; ++b){
        if (x < 0 || y < 0){
            i = rand() * py.size % h;
            j = rand() * py.size % h;
        }
        else{
            i = x*l.size;
            j = y*l.size;
        }
        for (k = 0; k < c; ++k){
            int in_index = j + w*i + k*h*w + b*c*h*w;
            for (n = -pad; n < py.size + pad; ++n){
                for (m = -pad; m < py.size + pad; ++m){
                    out = (m + pad) + (n + pad)*(py.size * 2) + k*(py.size*py.size * 2 * 2);
                    in = in_index + n*w + m;
                    if (in < 0 || in > l.inputs){
                        py.output[out] = 0;
                    }
                    else{
                        py.output[out] = incpu[in];
                    }
                }
            }
        }
        int truth_index = get_truth_index(level, l.size, i, j);
        cuda_push_array(py.output_gpu, py.output, py.batch*py.outputs);
        state.input = py.output_gpu;
        if (py.delta_gpu){
            fill_ongpu(py.outputs * py.batch, 0, py.delta_gpu, 1);
        }
        forward_network_pyramid_gpu(state.net, state, now , truth_index, level);
        backward_network_pyramid_gpu(state.net, state, now);
        if (l.level > 0){
            cuda_pull_array(py.delta_gpu, py.delta, py.batch*py.outputs);
            for (k = 0; k < c; ++k){
                int in_index = j + w*(i + h*(k + c*b));
                for (n = -pad; n < py.size + pad; ++n){
                    for (m = -pad; m < py.size + pad; ++m){
                        out = (m + pad) + (n + pad)*(py.size * 2) + k*(py.size*py.size * 2 * 2);
                        in = in_index + n*w + m;
                        if (in>=0){
                            delta[in] = py.delta[out];
                        }
                    }
                }
            }
        }
    }
}

void backward_pyramidpool_layer(const layer l, network_state state)
{
    int i;
    int h = (l.h - 1) / l.stride + 1;
    int w = (l.w - 1) / l.stride + 1;
    int c = l.c;
    for (i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}

void forward_pyramidpool_layer_test(float * incpu, layer l, layer py, network_state state, int now)
{
    int b, i, j, k, m, n;
    int h, w, c, th, level = 0, in, out;
    int pad = py.pad;

    if (l.type == PYRAMIDPOOL){
        h = l.h;
        w = l.w;
        c = l.c;
    }
    else{
        h = l.out_h;
        w = l.out_w;
        c = l.out_c;
    }

    th = h / 2;
    while (th>1){
        th /= 2;
        level++;
    }

    state.index = now;
    for (b = 0; b < l.batch; ++b){
        for (i = 0; i < h; i += py.size){
            for (j = 0; j < w; j += py.size){
                for (k = 0; k < c; ++k){
                    int in_index = j + w*(i + h*(k + c*b));
                    for (n = -pad; n < py.size + pad; ++n){
                        for (m = -pad; m < py.size + pad; ++m){
                            out = (m + pad) + (n + pad)*(py.size * 2) + k*(py.size*py.size * 2 * 2);
                            in = in_index + n*w + m;
                            if (in < 0 || in >= l.inputs){
                                py.output[out] = 0;
                            }
                            else{
                                py.output[out] = incpu[in];
                            }
                        }
                    }
                }
                int truth_index = get_truth_index(level, l.size, i, j);
                cuda_push_array(py.output_gpu, py.output, py.batch*py.outputs);
                state.input = py.output_gpu;
                forward_network_pyramid_gpu(state.net, state, now, truth_index, level);
            }
        }
    }
}

#ifdef GPU

void forward_pyramidpool_layer_gpu(layer l, network_state state, int i)
{
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);

    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.input = in_cpu;

    if (!state.train){
        layer lm = l;
        forward_pyramidpool_layer_test(in_cpu, lm, l, cpu_state, i + 1);
        for (int j = 1; j < l.level; j++){
            lm = state.net.pyramid[j - 1];
            forward_maxpool_layer_gpu(lm, state);

            state.input = lm.output_gpu;
            cuda_pull_array(state.input, in_cpu, lm.batch*lm.outputs);
            forward_pyramidpool_layer_test(in_cpu, lm, l, cpu_state, i + 1);
        }
        lm = state.net.layers[state.net.n - 1];
        cuda_push_array(lm.output_gpu, lm.output, lm.batch*lm.outputs);
        return;
    }
    
    float *delta_cpu = calloc(l.batch*l.inputs, sizeof(float));

    int num_truth = state.net.layers[state.net.n - 1].truths;
    float *truth_cpu = calloc(num_truth, sizeof(float));
    cuda_pull_array(state.truth, truth_cpu, num_truth);

    int truth_x[100] = { 0 }, truth_y[100] = { 0 }, level[100] = { 0 };
    get_truth_xy(truth_cpu, &num_truth, &truth_x, &truth_y, &level, l.level);
    free(truth_cpu);

    layer lm = l;
    for (int t = 0; t < num_truth; t++){
        if (level[t] == l.level -1){
            forward_pyramidpool_layer(in_cpu, lm, l, cpu_state, i + 1, delta_cpu, truth_x[t], truth_y[t]);
        }
    }
    //forward_pyramidpool_layer(in_cpu, lm, l, cpu_state, i + 1, delta_cpu, -1, -1);
    for (int j = 1; j < l.level; j++){
        state.input = lm.output_gpu;
        lm = state.net.pyramid[j - 1];
        forward_maxpool_layer_gpu(lm, state);

        state.input = lm.output_gpu;
        cuda_pull_array(state.input, in_cpu, lm.batch*lm.outputs);
        for (int t = 0; t < num_truth; t++){
            if (level[t] == (l.level - j -1)){
                forward_pyramidpool_layer(in_cpu, lm, l, cpu_state, i + 1, delta_cpu, truth_x[t], truth_y[t]);
            }
        }
        //forward_pyramidpool_layer(in_cpu, lm, l, cpu_state, i + 1, delta_cpu, -1, -1);
    }
    free(cpu_state.input);

    //cuda_push_array(l.delta_gpu, delta_cpu, l.batch*l.inputs);
    free(delta_cpu);
}

void backward_pyramidpool_layer_gpu(layer l, network_state state){
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}

#endif