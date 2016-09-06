#include "network.h"
#include "pyramid_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

char *perdestrian_names[] = {"person"};
image voc_labels[1];

void train_pyramid(char *cfgfile, char *weightfile)
{
    char *train_images = "E:/Experiment/YOLO/yolo_train/VOCdevkit/Caltech/train.txt";
    char *backup_directory = "E:/Experiment/YOLO/yolo_train/backup";
    char *log_file = "E:/Experiment/YOLO/yolo_train/backup/log.csv";
    FILE *log = fopen(log_file, "w");
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int level = l.level;
    int classes = l.classes;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.size = l.n;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.level = level;
    args.d = &buffer;
    args.type = PYRAMID_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        if (log){
            fprintf(log,"%d,%.5f\n", i, loss);
            fflush(log);
        }

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if( i%100 == 0 ){
            char buff[256];
            sprintf(buff, "%s/pyramid-%d.weights", backup_directory, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/pyramid-final.weights", backup_directory);
    save_weights(net, buff);
}

void convert_pyramid_detections(float *predictions, int level, int num, int square, int side, int w, int h, float thresh, float *probs, box *boxes)
{
    int i, j, c, n, k = 0, p_index, box_index, row;
    box b = { 0 };
    float step = 1.0 / num;
    for (i = 0; i < level; ++i){
        row = pow(2, i);
        for (j = 0; j < row; ++j){
            for (c = 0; c < row; ++c){
                for (n = 0; n < num; ++n){
                    box_index = k * num + n + j * row * num + c * num;
                    p_index = box_index * 6;
                    float prob = predictions[p_index] * predictions[p_index+1];
                    //float prob = predictions[p_index];
                    b.x = predictions[p_index + 2];
                    b.y = predictions[p_index + 3];
                    b.w = predictions[p_index + 4];
                    b.h = predictions[p_index + 5];

                    b.x = b.x * step;
                    b.y = constrain(0, 1, b.y);
                    b.h = b.h / 2.0;
                    b.h = b.h + 0.5;
                    b.w = b.h / 3.2;
                    if (prob > thresh){
                        probs[box_index] = prob;
                        //boxes[box_index].x = (b.x + 0.5 + j) / row ;
                        //boxes[box_index].y = (b.y + 0.5 + c) / row ;
                        //b.w = b.w > (0.5 / row) ? (0.5 / row) : b.w;
                        //b.h = b.h > (0.5 / row) ? (0.5 / row) : b.h;
                        //boxes[box_index].w = b.w + 0.5 / row ;
                        //boxes[box_index].h = b.h + 0.5 / row ;

                        boxes[box_index].x = (b.x + c + step*n ) / row ;
                        boxes[box_index].y = (b.y + j ) / row;
                        boxes[box_index].w = b.w / row ;
                        boxes[box_index].h = b.h / row ;

                        boxes[box_index].x *= w;
                        boxes[box_index].y *= h;
                        boxes[box_index].w *= w;
                        boxes[box_index].h *= h;
                    }
                }   
            }
        }
        k += pow(2, 2 * i);
    }
}

void print_pyramid_detections(FILE **fps, int id, box *boxes, float *probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        for(j = 0; j < classes; ++j){
            if (probs[i]) fprintf(fps[j], "%d %.2f %.2f %.2f %.2f %.2f\n", id, boxes[i].x - boxes[i].w / 2, boxes[i].y - boxes[i].h / 2, boxes[i].w, boxes[i].h, probs[i] * 100);
        }
    }
    for (i = 0; i < total; ++i){
        for (j = 0; j < classes; ++j){
            probs[i] = 0.0;
            boxes[i].x = 0.0;
            boxes[i].y = 0.0;
            boxes[i].w = 0.0;
            boxes[i].h = 0.0;
        }
    }
}

void validate_pyramid(char *cfgfile, char *weightfile, float thresh)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "data/";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("E:/Experiment/YOLO/yolo_train/VOCdevkit/Caltech/test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int level = l.level;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
		_snprintf(buff, 1024, "%s%s.txt", base, perdestrian_names[j]);
        fps[j] = fopen(buff, "w");
    }
    int k = 0;
    for (int i = 0; i < l.level; i++){
        k += pow(2, 2 * i);
    }
    box *boxes = calloc(k*l.n + 1, sizeof(box));
    float *probs = calloc(k*l.n + 1, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_pyramid_detections(predictions, l.level, l.n, l.sqrt, l.side, w, h, thresh, probs, boxes, 0);
            //if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_pyramid_detections(fps, (i + t - 1), boxes, probs, k*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_pyramid_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
		_snprintf(buff, 1024, "%s%s.txt", base, perdestrian_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
        convert_pyramid_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void test_pyramid(char *cfgfile, char *weightfile, char *filename, float thresh)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    pyramid_layer l = net.layers[net.n-1];
    l.classes = 1;
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    int k = 0;
    for (int i = 0; i < l.level; i++){
        k += pow(2, 2 * i);
    }
    box *boxes = calloc(k*l.n +1 , sizeof(box));
    float *probs = calloc(k*l.n +1 , sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_pyramid_detections(predictions, l.level, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        //if (nms) do_nms_sort(boxes, probs, k*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
		draw_detections(im, k*l.n, thresh, boxes, probs, perdestrian_names, voc_labels, l.classes);
        save_image(im, "predictions");
        show_image(im, "predictions");

        //show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void truth_pyramid(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    char *train_images = "E:/Experiment/YOLO/yolo_train/VOCdevkit/Caltech/train.txt";
    char *backup_directory = "E:/Experiment/YOLO/yolo_train/backup";
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if (weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data  buffer;
    layer l = net.layers[net.n - 1];
    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);

    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    int j;
    int k = 0;
    for (int i = 0; i < l.level; i++){
        k += pow(2, 2 * i);
    }
    box *boxes = calloc(k*l.n + 1, sizeof(box));
    float *probs = calloc(k*l.n + 1, sizeof(float *));
    while (1){
        time = clock();
        buffer = load_data_pyramid(1, paths, plist->size, net.w, net.h, l.level, 0, l.n);

        printf("Loaded: %lf seconds\n", sec(clock() - time));
        char *input = buffer.fname;

        image im = load_image_color(input, 0, 0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time = clock();
        //float loss = train_network(net, buffer);
        float *predictions = buffer.y.vals[0];

        printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
        convert_pyramid_detections(predictions, l.level, 1, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);

        draw_detections(im, k*l.n, thresh, boxes, probs, perdestrian_names, voc_labels, l.classes);
        show_image(im, "predictions");
        show_image(sized, "resized");

        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void run_pyramid(int argc, char **argv)
{
    int i;
    for(i = 0; i < 1; ++i){
        char buff[256];
        //sprintf(buff, "data/labels/%s.png", voc_names[i]);
		sprintf(buff, "E:/Experiment/YOLO/darknet/data/labels/%s.png", perdestrian_names[i]);
        voc_labels[i] = load_image_color(buff, 0, 0);
    }

	float thresh = find_float_arg(argc, argv, "-thresh", .2);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
	int frame_skip = find_int_arg(argc, argv, "-s", 0);
	if (argc < 4){
		fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
		return;
	}

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_pyramid(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_pyramid(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_pyramid(cfg, weights, thresh);
    else if(0==strcmp(argv[2], "recall")) validate_pyramid_recall(cfg, weights);
	else if (0 == strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, perdestrian_names, voc_labels, 1, frame_skip);
    else if (0 == strcmp(argv[2], "truth")) truth_pyramid(cfg, weights, filename, thresh);
}
