#include "fast_dot_prod.h"
#include <iostream>

#define FAST_DOT

//#undef USE_INTRIN
#ifdef USE_INTRIN
#include "intrinsic_help.h"
float fast_dot_prod(const float * buf1,const float * buf2,size_t size){
    constexpr size_t BUF_SIZE = 32;
    constexpr size_t VS = 8;
    fvec8 vec1,vec2,vec3,vec4;
    size_t i = 0;
    for(; i <= size-BUF_SIZE; i += BUF_SIZE){
        vec1 += fvec8(buf1+i+0*VS) * fvec8(buf2+i+0*VS);
        vec2 += fvec8(buf1+i+1*VS) * fvec8(buf2+i+1*VS);
        vec3 += fvec8(buf1+i+2*VS) * fvec8(buf2+i+2*VS);
        vec4 += fvec8(buf1+i+3*VS) * fvec8(buf2+i+3*VS);
    }
    if(i <= size-VS){
        vec1 += fvec8(buf1+i) * fvec8(buf2+i);
        i += VS;
        if(i <= size-VS){
            vec2 += fvec8(buf1+i) * fvec8(buf2+i);
            i += VS;
            if(i <= size-VS){
                vec3 += fvec8(buf1+i) * fvec8(buf2+i);
                i += VS;
            }
        }
    }
    vec1 += vec2 + vec3 + vec4;

    float sum = vec1.sum();
    for(; i < size; i += 1){
        sum += buf1[i] * buf2[i];
    }
    return sum;
}

#elif defined FAST_DOT
float fast_dot_prod(const float * buf1,const float * buf2,size_t size){
    constexpr size_t BUF_SIZE = 32;
    float sumbuf[BUF_SIZE] = {0};
    size_t i = 0;
    for(; i <= size-BUF_SIZE; i += BUF_SIZE){
        for(size_t j = 0; j < BUF_SIZE; j++){
            sumbuf[j] += buf1[i+j] * buf2[i+j];
        }
    }
    for(size_t j = 0; j < BUF_SIZE/2; j++){
        sumbuf[j] += sumbuf[j + BUF_SIZE/2];
    }
    for(size_t j = 0; j < BUF_SIZE/4; j++){
        sumbuf[j] += sumbuf[j + BUF_SIZE/4];
    }
    float sum = 0;
    for(size_t j = 0; j < BUF_SIZE/4; j++){
        sum += sumbuf[j];
    }
    float sum2 = 0;
    for(; i < size; i++){
        sum2 += buf1[i] * buf2[i];
    }

    return sum + sum2;
}
#else
float fast_dot_prod(const float * buf1,const float * buf2,size_t size){
    float sum = 0;
    for(size_t i = 0; i < size; i++){
        sum += buf1[i] * buf2[i];
    }
    return sum;
}
#endif
