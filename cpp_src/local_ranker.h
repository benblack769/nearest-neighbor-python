#pragma once
#include "types.h"
#include <vector>
#include <cstddef>


size_t sigbit_index(uint64_t pos);
class LocalRanker{
private:
    std::vector<std::vector<pos_ty>> vec;
public:
    void add_weight(pos_ty source,pos_ty dest,float weight){

    }
    pos_ty sample_local(pos_ty source){

    }
    bool can_sample(pos_ty source){

    }
    void swap_idxs(pos_ty idx1, pos_ty idx2){

    }
};
