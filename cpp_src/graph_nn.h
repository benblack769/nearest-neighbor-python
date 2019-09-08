#pragma once
#include <vector>
#include <string>

struct GraphAccessor;
using pos_id = uint64_t;
using vecf =  std::vector<float>;
GraphAccessor * create_graph_accessor(const char * accessor_path,int position_size);
void free_graph_accessor(GraphAccessor * ba);
void add_all_positions(GraphAccessor * ba,std::vector<std::string> npy_filenames);
//endline seperated list of ids
std::vector<pos_id> fetch_similar(GraphAccessor * ba,const float * position, float max_distance, int max_count);
vecf fetch_vec(GraphAccessor * ba,pos_id id);
