#pragma once
#include "types.h"
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cassert>
#include "unif_float_sampler.h"


class Ranker{
    using value_ty = uint32_t;
    static constexpr size_t NO_IDX = ~size_t(0);
    static constexpr value_ty NO_RANK = ~value_ty(0);
    static constexpr pos_ty NO_POS = ~pos_ty(0);
    std::vector<value_ty> values;
    std::vector<pos_ty> ids;
    size_t max_size;
    size_t cur_dec = 0;
    value_ty total_value = 0;
public:
    Ranker(size_t in_max_size){
        values.reserve(in_max_size);
        ids.reserve(in_max_size);
        max_size = in_max_size;
    }
    size_t size(){
        return values.size();
    }
    pos_ty sample(UnifFloatSampler & sampler){
        size_t size = values.size();
        assert(size > 0);
        value_ty target = sampler.sample() * total_value;
        value_ty count_val = 0;
        for(size_t i = 0; i < size; i++){
            count_val += values[i];
            if(count_val < target){
                return ids[i];
            }
        }
        return ids.back();
    }
    void increment(pos_ty pos){
        size_t idx = NO_IDX;
        size_t size = values.size();
        for(size_t i = 0; i < size; i++){
            if(pos == ids[i]){
                idx = i;
                break;
            }
        }
        if(idx != NO_IDX){
            values[idx]++;
            total_value++;
        }
        else{
            if(values.size() < max_size){
                ids.push_back(pos);
                values.push_back(1);
                total_value++;
            }
            else{
                if(cur_dec >= max_size){
                    cur_dec = 0;
                }
                //
                values[cur_dec]--;
                total_value--;
                if(values[cur_dec] == 0){
                    ids[cur_dec] = pos;
                    values[cur_dec] = 1;
                    total_value++;
                }
                cur_dec++;
            }
        }
    }
    void resize(size_t new_size){
        if(new_size > max_size){
            grow(new_size);
        }
        else if(new_size < max_size){
            if(new_size < values.size()){
                shrink(new_size);
            }
            else{
                values.shrink_to_fit();
                ids.shrink_to_fit();

                values.reserve(new_size);
                ids.reserve(new_size);

                max_size = new_size;
            }
        }
    }
private:
    void recount_total(){
        total_value = std::accumulate(values.begin(),values.end(),0);
    }
    void shrink(size_t new_size){
        struct PosVal{value_ty val; pos_ty pos;};
        std::vector<PosVal> posses(values.size());
        for(size_t i = 0; i < values.size(); i++){
            posses[i] =  PosVal{.val=values[i],.pos=ids[i]};
        }
        std::sort(posses.begin(),posses.end(),[](const PosVal & v1,const PosVal & v2){
            return v1.val > v2.val;
        });
        values.resize(new_size);
        values.shrink_to_fit();
        ids.resize(new_size);
        ids.shrink_to_fit();
        for(size_t i = 0; i < new_size; i++){
            values[i] = posses[i].val;
            ids[i] = posses[i].pos;
        }
        recount_total();
    }
    void grow(size_t new_size){
        ids.reserve(new_size);
        values.reserve(new_size);
        max_size = new_size;
    }
};
size_t sigbit_index(uint64_t pos);
class LocalRanker{
private:
    std::vector<Ranker> vec;
    UnifFloatSampler sampler;
public:
    void add_node(){
        size_t max_size = 32;
        vec.emplace_back(max_size);
    }
    void inc_weight(pos_ty source,pos_ty dest){
        vec.at(source).increment(dest);
    }
    pos_ty sample_local(pos_ty source){
        assert(can_sample(source));
        vec.at(source).sample(sampler);
    }
    bool can_sample(pos_ty source){
        return vec.at(source).size() > 0;
    }
    //void swap_idxs(pos_ty idx1, pos_ty idx2){
    //    vec.at(idx1).swap(vec.at(idx2));
    //}
};
