#pragma once
#include <vector>
#include <random>
#include <cassert>
#include <stdexcept>
#include "types.h"
#include "unif_float_sampler.h"

class GlobalRanker{
private:
    using val_ty = uint64_t;
    std::vector<val_ty> cumsum_weight;
    std::vector<val_ty> value_weights;
    //std::vector<pos_ty> value_idxs;

    std::mt19937 e2;

    void set_weight(pos_ty pos,val_ty weight){
        val_ty diff = weight - value_weights.at(pos);
        value_weights.at(pos) = weight;
        while(pos > 0){
            cumsum_weight.at(pos) += diff;
            pos = (pos - 1) / 2;
        }
        cumsum_weight.at(0) += diff;
    }
public:
    GlobalRanker() = default;
    void add_weight(pos_ty pos,val_ty weight){
        set_weight(pos,this->weight(pos)+weight);
    }
    size_t size(){
        return value_weights.size();
    }
    void append_value(float weight){
        size_t loc = value_weights.size();
        value_weights.push_back(0);
        cumsum_weight.push_back(0);
        set_weight(loc,weight);
    }
    bool can_sample(){
        return value_weights.size() != 0;
    }
    pos_ty sample(){
        assert(cumsum_weight.size() > 0);
        val_ty total_val = cumsum_weight.front();
        std::uniform_int_distribution<val_ty> dist(0,total_val);

        val_ty sample = dist(e2);
        pos_ty pos = 0;

        val_ty cur_val = 0;
        while(pos*2+2 < cumsum_weight.size()){
            if(sample <= cur_val + cumsum_weight[pos*2+1]){
                pos = pos*2+1;
            }
            else{
                cur_val += cumsum_weight[pos*2+1];
                if(sample <= cur_val + cumsum_weight[pos*2+2]){
                    pos = pos*2+2;
                }
                else{
                    return pos;
                }
            }
        }
        if(pos*2+1 < cumsum_weight.size()){
            if(sample <= cur_val + cumsum_weight[pos*2+1]){
                return pos*2+1;
            }
            else{
                return pos;
            }
        }
        else{
            return pos;
        }
        throw std::runtime_error(std::to_string(pos));
    }
    val_ty weight(pos_ty idx){
        return value_weights.at(idx);
    }
    /*std::vector<pos_ty> get_best_idxs(int count){
        return std::vector<pos_ty>({1,2,3,4,5,6,7});
    }*/
    void swap_idxs(pos_ty idx1, pos_ty idx2){
        val_ty weight1 = value_weights.at(idx1);
        val_ty weight2 = value_weights.at(idx2);
        set_weight(idx1,weight2);
        set_weight(idx2,weight1);
    }
};
