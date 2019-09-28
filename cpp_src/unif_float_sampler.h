#pragma once
#include <random>

class UnifFloatSampler{
private:
    std::mt19937 e2;
    std::uniform_real_distribution<float> dist;
public:
    UnifFloatSampler():
        e2(),
        dist(0.0f,1.0f){}
    float sample(){
        return dist(e2);
    }
};
