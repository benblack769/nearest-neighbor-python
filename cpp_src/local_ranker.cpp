#include "local_ranker.h"
constexpr uint64_t mask_for_shift(int shift_val){
    uint64_t mask = ~uint64_t(0);
    return ~(mask << shift_val);
}
template<int shift_val>
inline size_t sigbit_index_r(uint64_t pos){
    constexpr uint64_t mask = mask_for_shift(shift_val);
    uint64_t rec_val;
    size_t add_val;
    if(pos & (mask << shift_val)){
        rec_val = (pos & (mask << shift_val)) >> shift_val;
        add_val = shift_val;
    }
    else{
        rec_val = (pos & mask);
        add_val = 0;
    }
    return add_val + sigbit_index_r<shift_val/2>(rec_val);
}
template<>
inline size_t sigbit_index_r<0>(uint64_t pos){
    return 0;
}
size_t sigbit_index(uint64_t pos){
    return sigbit_index_r<32>(pos);
}
