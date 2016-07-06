#ifndef PYRAMID_LAYER_H
#define PYRAMID_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer pyramid_layer;

pyramid_layer make_pyramid_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);
void forward_pyramid_layer(const pyramid_layer l, network_state state);
void backward_pyramid_layer(const pyramid_layer l, network_state state);

#ifdef GPU
void forward_pyramid_layer_gpu(const pyramid_layer l, network_state state);
void backward_pyramid_layer_gpu(pyramid_layer l, network_state state);
#endif

#endif
