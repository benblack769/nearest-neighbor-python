#pragma once
#include <cstdint>

using pos_id = uint64_t;
using pos_ty = uint64_t;

struct ValIdPair{
    pos_id id;
    float val;
};

struct ValPosPair{
    pos_ty idx;
    float val;
};
