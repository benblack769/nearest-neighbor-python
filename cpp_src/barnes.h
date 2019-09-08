#pragma once
#include <vector>
#include <string>
constexpr int MAX_ID_SIZE = 128-1;
struct BarnesAccessor;
using pos_id = uint64_t;
using vecf =  std::vector<float>;
BarnesAccessor * create_barnes_accessor(const char * accessor_path,int position_size);
void free_barnes_accessor(BarnesAccessor * ba);
void add_position(BarnesAccessor * ba,pos_id id,const float * position);
void update_position(BarnesAccessor * ba,pos_id id,const float * position);
//endline seperated list of ids
std::vector<pos_id> fetch_similar(BarnesAccessor * ba,const float * position, float max_distance, int max_count);
vecf fetch_vec(BarnesAccessor * ba,pos_id id);
