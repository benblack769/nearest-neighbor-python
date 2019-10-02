#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "types.h"
#include "unif_float_sampler.h"

struct PosVal{
    uint64_t pos;
    float val;
    bool operator < (const PosVal & other)const{
        return val > other.val;
    }
};
class SampleableMinHeap{
    std::vector<PosVal> data;
public:
    void push(PosVal posval){
        data.push_back(posval);
        std::push_heap(data.begin(),data.end());
    }
    PosVal min(){
        return data.front();
    }
    void pop(){
        std::pop_heap(data.begin(),data.end());
        data.pop_back();
    }
    size_t size(){
        return data.size();
    }
    pos_ty sample(UnifIntSampler & sampler){
        size_t rank_size = data.size();
        size_t item_sample = sampler.sample(rank_size);
        return data[item_sample].pos;
    }
};

class RankSampler{
public:
    using value_ty = float;
private:
    std::vector<SampleableMinHeap> values;
public:
    pos_ty sample(UnifIntSampler & sampler){
        int num_ranks = std::max(size_t(1),values.size()-1);
        int rank_sample = sampler.sample(num_ranks);
        return values.at(rank_sample).sample(sampler);
    }
    bool can_sample(){
        return values.size() > 0;
    }
    void add(pos_ty pos,value_ty val){
        if(!values.size() || values.back().size() >= rank_max(values.size())){
            values.emplace_back();
        }
        size_t rank = 0;
        for(; rank < values.size()-1; rank++){
            if(values[rank].min().val < val){
                break;
            }
        }
        PosVal cur_pval = {.pos=pos,.val=val};
        for(;rank < values.size()-1; rank++){
            PosVal swap_out = values[rank].min();
            values[rank].pop();
            values[rank].push(cur_pval);
            cur_pval = swap_out;
        }
        values.back().push(cur_pval);
    }
private:
    size_t rank_max(int rank){
        return size_t(1)<<size_t(rank);
    }
};
