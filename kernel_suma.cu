#include <stdio.h>

#define image_channels 4

__device__ void process_pixel(unsigned char* src_image,unsigned char* src_image1, unsigned char* dst_image, int width, int height, int x, int y, int blur_step) {

    int index = (y * width + x)*image_channels;
    dst_image[index+0] = (src_image[index]+src_image1[index])/2;
    dst_image[index+1] = (src_image[index+1]+src_image1[index+1])/2;
    dst_image[index+2] = (src_image[index+2]+src_image1[index+2])/2;
    dst_image[index+3] = 255;
}

__global__ void kernel_add_image(unsigned char* src_image,unsigned char* src_image1, unsigned char* dst_image, int width, int height, int blur_step) {

    int index = blockDim.x*blockIdx.x + threadIdx.x;

    int pix_y = index / width;
    int pix_x = index % width;

    process_pixel(src_image,src_image1, dst_image, width, height, pix_x,pix_y, blur_step);
}

extern "C" void kernel_add(unsigned char* src_image,unsigned char* src_image1, unsigned char* dst_image, int width, int height, int blur_step) {
    //kernel
    int thr_per_blk = 1024;//256;
    int blk_in_grid = ceil( float(width*height) / thr_per_blk );

    kernel_add_image<<<blk_in_grid,thr_per_blk>>>(src_image,src_image1, dst_image, width, height, blur_step);
}