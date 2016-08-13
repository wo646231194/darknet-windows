#ifndef PYRAMIDPOOL_LAYER_H
#define PYRAMIDPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "pyramid_layer.h"

typedef struct pyramidpool_layer
{
    layer layer;
    maxpool_layer pool[5];
    int type;
}pyramidpool_layer;

image get_pyramidpool_image(pyramidpool_layer l);
pyramidpool_layer make_pyramidpool_layer(int batch, int h, int w, int c, int level, int size, int pad);
void resize_pyramidpool_layer(pyramidpool_layer *l, int w, int h);
void forward_pyramidpool_layer(float * incpu, layer l, pyramidpool_layer py, network_state state, int now, float *delta, int x, int y);
void backward_pyramidpool_layer(const layer l, network_state state);

#ifdef GPU
void forward_pyramidpool_layer_gpu(layer l, network_state state, int i);
void backward_pyramidpool_layer_gpu(layer l, network_state state);
#endif

#endif

