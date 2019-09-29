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
class UnifIntSampler{
private:
    std::mt19937 e2;
public:
    UnifIntSampler():
        e2(){}
    uint64_t sample(uint64_t max){
        std::uniform_int_distribution<uint64_t> dist(0,max);
        return dist(e2);
    }
};
