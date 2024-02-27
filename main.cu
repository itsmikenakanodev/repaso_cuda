#include <iostream>
#include <vector>
#include <chrono>
#include <fmt/core.h>
#include <random>
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>

#define VECTOR_ELEMENTS 5000000
#define WIDTH 1280
#define HEIGHT 720

__global__ void min_kernel(int *input,int n, int *min) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;

    if(index<n){
        atomicMin(min,input[index]);
    }


}

__global__ void max_kernel(int *input,int n, int *min) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;

    if(index<n){
        atomicMax(min,input[index]);
    }

}

__global__ void promedio_kernel(int *input,int n, int *suma) {
    int index=blockDim.x*blockIdx.x+threadIdx.x;

    if(index<n){
        atomicAdd(suma,input[index]);
    }
}

__global__ void Kogge_Stone_scan_kernel(int *X, int *Y, int n) {
    __shared__ int XY[256];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && threadIdx.x != 0) {
        XY[threadIdx.x] = X[i-1];
    } else {
        XY[threadIdx.x] = 0;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride*= 2) {
        __syncthreads();
        if (threadIdx.x >= stride) XY[threadIdx.x] +=XY[threadIdx.x-stride];
    }
    Y[i] = XY[threadIdx.x];
}

__global__ void Brent_Kung_scan_kernel(int *X, int*Y,
                                       int n) {
    __shared__ int XY[256];
    int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) XY[threadIdx.x] = X[i];
    if (i+blockDim.x < n) XY[threadIdx.x+blockDim.x] = X[i+blockDim.x];
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x+1) * 2* stride -1;
        if (index < 256) {
            XY[index] += XY[index - stride];
        }
    }
    for (int stride = 256/4; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index + stride < 256) {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < n) Y[i] = XY[threadIdx.x];
    if (i+blockDim.x < n) Y[i+blockDim.x] = XY[threadIdx.x+blockDim.x];
}

__global__ void histograma_kernel(int * input,int n, int* histo, int block_histo){
    int index = threadIdx.x + blockIdx.x * blockDim.x; //   255  +   (2*256) = 2        id=767     768-total de hilos que tenemos
    int section_size=(n-1)/(blockDim.x*gridDim.x) +1;  //  99/256*1  +1 =  1.12
    int bloques = 256/block_histo;
    int start= index*section_size; // 255*1 =510

    for(int i=0;i<section_size;i++){
        if(start+i<n){
            int numero_pos=input[start+i];
            if(numero_pos >=0 && numero_pos<256){
                atomicAdd(&(histo[numero_pos/bloques]),1);
            }
        }
    }
}

__global__ void histograma_kernel1(int * input,int n, int* histo, int block_histo){
    int index = threadIdx.x + blockIdx.x * blockDim.x; //   255  +   (2*256) = 2        id=767     768-total de hilos que tenemos
    //int section_size=(n-1)/(blockDim.x*gridDim.x) +1;  //  99/256*1  +1 =  1.12
    int bloques = 256/block_histo;
    //int start= index*section_size; // 255*1 =510

    if(index<n){
        int numero_pos=input[index];
        if(numero_pos >=0 && numero_pos<256){
                atomicAdd(&(histo[numero_pos/bloques]),1);
        }
    }

}

static std::vector<int> initVectorHist(const int size){
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uniform_dist(0, 255);
    std::vector<int>res;
    res.resize(size);
    for (int i = 0; i < size; ++i) {
        res[i]=uniform_dist(rng);
    }
    return res;
}

void prefixSum(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    output.resize(n);

    int* d_input;
    int* d_output;

    // Se reserva memoria en el device para los vectores de entrada y salida
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));

    // Se copia el vector de entrada desde la CPU al GPU
    cudaMemcpy(d_input, input.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int blockDim = 128;
    int gridDim = (n + 2 * blockDim - 1) / (2 * blockDim);

    // Se llama al kernel para realizar el escaneo de Brent-Kung en paralelo en el GPU
    Brent_Kung_scan_kernel<<<gridDim, blockDim>>>(d_input, d_output, n);

    // Se copia el resultado de la suma prefija desde el GPU a la CPU
    cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Se libera la memoria reservada en el device
    cudaFree(d_input);
    cudaFree(d_output);
}

std::vector<int> histograma(std::vector<int> input, int bloques){
    int block_histo = bloques;
    std::vector<int> vector=input;
    int * d_vector;

    std::vector<int> h_histograma(block_histo);
    int * d_histograma;

    // Se reserva memoria en la GPU para el vector de entrada y el histograma
    cudaMalloc(&d_vector,VECTOR_ELEMENTS * sizeof (int));
    cudaMalloc(&d_histograma, block_histo * sizeof (int));

    // Se copia el vector de entrada desde la CPU al GPU
    cudaMemcpy(d_vector, vector.data(), VECTOR_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

    int block_size=256;
    if(VECTOR_ELEMENTS<256){
        block_size=VECTOR_ELEMENTS;
    }
    int grid_size =  std::ceil((double) VECTOR_ELEMENTS / block_size);
    fmt::println("Numero de hilos: {}",block_size);
    fmt::println("Numero de bloques: {}",grid_size);
    //grid dim -> hace referencia al numero de bloques
    //block dim -> numero de hilos

    // Se llama al kernel para calcular el histograma en paralelo en el GPU
    histograma_kernel1<<<grid_size, block_size>>>(d_vector,VECTOR_ELEMENTS,d_histograma,block_histo);

    // Se copia el resultado del histograma desde el GPU a la CPU
    cudaMemcpy(h_histograma.data(), d_histograma,  block_histo * sizeof(int), cudaMemcpyDeviceToHost);

    // Se libera la memoria reservada en el dispositivo
    cudaFree(d_histograma);
    cudaFree(d_vector);

    return h_histograma;
}

int calcular_min(std::vector<int> input){
    int * d_vector;

    int h_min=input[0];
    int* d_min;

    // Se reserva memoria en la GPU para el vector de entrada y el histograma
    cudaMalloc(&d_vector,input.size() * sizeof (int));
    cudaMalloc((void **)&d_min, sizeof(int));

    // Se copia el vector de entrada desde la CPU al GPU
    cudaMemcpy(d_vector, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice);

    int block_size=256;
    if(input.size()<256){
        block_size=input.size();
    }
    int grid_size =  std::ceil((double) input.size() / block_size);
    fmt::println("Numero de hilos: {}",block_size);
    fmt::println("Numero de bloques: {}",grid_size);
    //grid dim -> hace referencia al numero de bloques
    //block dim -> numero de hilos

    min_kernel<<<grid_size, block_size>>>(d_vector,input.size(),d_min);

    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

    // Se libera la memoria reservada en el dispositivo
    cudaFree(d_min);
    cudaFree(d_vector);

    return h_min;
}

int calcular_max(std::vector<int> input){
    int * d_vector;

    int h_max=input[0];
    int* d_max;

    // Se reserva memoria en la GPU para el vector de entrada y el histograma
    cudaMalloc(&d_vector,input.size() * sizeof (int));
    cudaMalloc((void **)&d_max, sizeof(int));

    // Se copia el vector de entrada desde la CPU al GPU
    cudaMemcpy(d_vector, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice);

    int block_size=256;
    if(input.size()<256){
        block_size=input.size();
    }
    int grid_size =  std::ceil((double) input.size() / block_size);
    fmt::println("Numero de hilos: {}",block_size);
    fmt::println("Numero de bloques: {}",grid_size);
    //grid dim -> hace referencia al numero de bloques
    //block dim -> numero de hilos

    max_kernel<<<grid_size, block_size>>>(d_vector,input.size(),d_max);

    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    // Se libera la memoria reservada en el dispositivo
    cudaFree(d_max);
    cudaFree(d_vector);

    return h_max;
}

double calcular_promedio(std::vector<int> input){
    int * d_vector;

    int h_suma=0;
    int* d_suma;

    // Se reserva memoria en la GPU para el vector de entrada y el histograma
    cudaMalloc(&d_vector,input.size() * sizeof (int));
    cudaMalloc((void **)&d_suma, sizeof(int));

    // Se copia el vector de entrada desde la CPU al GPU
    cudaMemcpy(d_vector, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suma, &h_suma, sizeof(int), cudaMemcpyHostToDevice);

    int block_size=256;
    if(input.size()<256){
        block_size=input.size();
    }
    int grid_size =  std::ceil((double) input.size() / block_size);
    fmt::println("Numero de hilos: {}",block_size);
    fmt::println("Numero de bloques: {}",grid_size);
    //grid dim -> hace referencia al numero de bloques
    //block dim -> numero de hilos

    promedio_kernel<<<grid_size, block_size>>>(d_vector,input.size(),d_suma);

    cudaMemcpy(&h_suma, d_suma, sizeof(int), cudaMemcpyDeviceToHost);

    // Se libera la memoria reservada en el dispositivo
    cudaFree(d_suma);
    cudaFree(d_vector);

    return (double)h_suma/input.size();
}


int max(std::vector<int>val){
    int max=0;
    for (int i = 0; i < val.size(); ++i) {
        if (max<val[i]){
            max=val[i];
        }
    }
    fmt::println("MAXIMO {}",max);
    return max;
}

void renderHisto(std::vector<int> values){
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Histograma");

    int maximo=max(values);
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        window.clear(sf::Color::Black);
        for (int i = 0; i < values.size(); ++i) {
            sf::RectangleShape rectangle(sf::Vector2f((float) (WIDTH / values.size()) , -(values[i] * HEIGHT) / maximo));
            rectangle.setOutlineThickness(1.f);
            rectangle.setOutlineColor(sf::Color(250, 150, 100));
            rectangle.setPosition(sf::Vector2f((float) i * (WIDTH / values.size()), HEIGHT));
            window.draw(rectangle);
        }
        window.display();
    }
}

int main() {

    std::vector<int> input(VECTOR_ELEMENTS);
    for (int i = 0; i < input.size(); i++) {
        input[i]=i;
    }
    std::vector<int> output(input.size());
    prefixSum(input, output);
/*
    // Scan/Prefix Sum
    input.resize(VECTOR_ELEMENTS);
    for (int i = 0; i < input.size(); i++) {
        input[i]=i;
    }
    output.resize(input.size());
    auto start = std::chrono::high_resolution_clock::now();
    prefixSum(input, output);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tiempo = end - start;
    fmt::println("Prefix sum CUDA time: {} ms",tiempo.count());*/
//    for (int i = 0; i < output.size(); ++i) {
//        fmt::println("Indice: {} , Valor: {}", i, output[i] );
//    }

    ///Histograma
    std::vector<int> valores= initVectorHist(VECTOR_ELEMENTS);
    auto start1 = std::chrono::high_resolution_clock::now();
    std::vector<int>vector_histograma;
    vector_histograma=histograma(valores, 8);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tiempo1 = end1 - start1;
    fmt::println("Histograma CUDA time: {} ms",tiempo1.count());
    int suma=0;
    for (int i = 0; i < vector_histograma.size(); i++) {
        //fmt::println("Indice: {} , Valor: {}", i, vector_histograma[i] );
        suma +=vector_histograma[i];
    }
    fmt::println("Suma: {}", suma);
    /*renderHisto(vector_histograma);*/

    //min
   /* suma=0;
    for (int i = 0; i < valores.size(); i++) {
        fmt::println("Indice: {} , Valor: {}", i, valores[i] );
        suma +=valores[i];
    }
    fmt::println("Promedio: {}", (double )suma/valores.size());
    int min= calcular_min(valores);
    fmt::println("Minimo: {}", min);

    int max= calcular_max(valores);
    fmt::println("Maximo: {}", max);

    double promedio= calcular_promedio(valores);
    fmt::println("Promedio: {}", promedio);*/

    return 0;
}
