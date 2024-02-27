
#include <vector>
#include <fmt/core.h>
#include <random>

#ifndef GRUPAL_MAIN_2_UTILS_H
#define GRUPAL_UTILS_H



class utils {

public:


    static void printVector(const std::vector<int>& vec) {
        for (const auto& element : vec) {
            fmt::print("{} ", element);
        }
        fmt::print("\n");
    }
    static std::vector<int> initVector(const int size){
        std::vector<int>res(size);
        for (int i = 0; i < size; ++i) {
            res[i]=i;
        }
        return res;
    }
    static int* initVectorHist(const int size){
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uniform_dist(0, 255);
        int * res= new int[size];
        for (int i = 0; i < size; ++i) {
            res[i]=uniform_dist(rng);
        }
        return res;
    }

};


#endif //GRUPAL_UTILS_H
