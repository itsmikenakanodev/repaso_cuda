#include <stdio.h>

#define image_channels 4

__device__ void process_pixel(unsigned char* src_image, unsigned char* dst_image, int width, int height, int x, int y, int blur_step) {

    int index = (y * width + x)*image_channels;
    int intensidad = src_image[index]*0.299+src_image[index+1]*0.587+src_image[index+2]*0.114;
    dst_image[index+0] = intensidad;
    dst_image[index+1] = intensidad;
    dst_image[index+2] = intensidad;
    dst_image[index+3] = 255;
}

__global__ void kernel_gray_image(unsigned char* src_image, unsigned char* dst_image, int width, int height, int blur_step) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel(src_image, dst_image, width, height, pix_x,pix_y, blur_step);
}

extern "C" void kernel_gray(unsigned char* src_image, unsigned char* dst_image, int width, int height, int blur_step) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kernel_gray_image<<<blk_in_grid,thr_per_blk>>>(src_image, dst_image, width, height, blur_step);
}