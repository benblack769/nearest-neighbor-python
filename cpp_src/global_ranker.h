#pragma once
#include <vector>
#include <random>
#include "types.h"

class UnifSampler{
private:
    std::mt19937 e2;
    std::uniform_real_distribution<float> dist;
public:
    UnifSampler():
        e2(),
        dist(0.0f,1.0f){}
    float sample(){
        return dist(e2);
    }
};

class GlobalRanker{
private:
    std::vector<float> cumsum_weight;
    std::vector<float> value_weights;
    std::vector<pos_ty> value_idxs;
    UnifSampler sampler;
public:
    GlobalRanker() = default;
    void set_weight(pos_ty pos,float weight){

    }
    void append_value(float weight){

    }
    pos_ty sample(){
        float sample = sampler.sample();
    }
    float weight(pos_ty idx){
        return value_weights.at(idx);
    }
    std::vector<pos_ty> get_best_idxs(int count){
        return std::vector<pos_ty>({1,2,3,4,5,6,7});
    }
    void swap_idxs(pos_ty idx1, pos_ty idx2){

    }
};
