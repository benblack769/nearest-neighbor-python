#pragma once
#include <immintrin.h>
class fvec8
{
public:
	typedef float num_ty;

    explicit fvec8(float A,float B,float C,float D,float E,float F,float G,float H){
		d = _mm256_set_ps(A,B,C,D,E,F,G,H);
    }
    explicit fvec8(float num){
        d = _mm256_set1_ps(num);
    }
    explicit fvec8(const float * src){
        d = _mm256_loadu_ps(src);
    }
    fvec8(__m256 d_in){
        d = d_in;
    }
    fvec8(){
        d = _mm256_setzero_ps();
    }

    __m256 d;
    fvec8 & operator += (const fvec8 & other){
        d = _mm256_add_ps(d,other.d);
        return *this;
    }
    fvec8 & operator *= (const fvec8 & other){
        d = _mm256_mul_ps(d,other.d);
        return *this;
    }
    fvec8 & operator -= (const fvec8 & other){
        d = _mm256_sub_ps(d,other.d);
        return *this;
    }
    fvec8 & operator /= (const fvec8 & other){
        d = _mm256_div_ps(d,other.d);
        return *this;
    }
    fvec8 operator + (const fvec8 & other){
        return _mm256_add_ps(d,other.d);
    }
    fvec8 operator - (const fvec8 & other){
        return _mm256_sub_ps(d,other.d);
    }
    fvec8 operator * (const fvec8 & other){
        return _mm256_mul_ps(d,other.d);
    }
	fvec8 operator / (const fvec8 & other){
		return _mm256_div_ps(d,other.d);
	}
    fvec8 aprox_recip(){
        return _mm256_rcp_ps(d);
	}
    void store(float * dest){
        _mm256_storeu_ps(dest,d);
    }
    float sum(){
        float ds[8] = {0};
        store(ds);
        ds[0] += ds[4];
        ds[1] += ds[5];
        ds[2] += ds[6];
        ds[3] += ds[7];
        ds[0] += ds[2];
        ds[1] += ds[3];
        ds[0] += ds[1];
        return ds[0];
    }
	size_t size(){
		return 8;
	}
	float * begin(){
		return reinterpret_cast<float *>(this);
	}
	float * end(){
		return reinterpret_cast<float *>(this) + size();
	}
};
fvec8 fma (const fvec8 & x,const fvec8 & y,const fvec8 & z){
#ifdef __FMA__
	return _mm256_fmadd_ps(x.d,y.d,z.d);
#else
	return _mm256_add_ps(_mm256_mul_ps(x.d,y.d),z.d);
#endif
}
inline fvec8 max(const fvec8 & one,const fvec8 & two){
    return _mm256_max_ps(one.d,two.d);
}
inline fvec8 sqrt(const fvec8 & x){
    return _mm256_sqrt_ps(x.d);
}
