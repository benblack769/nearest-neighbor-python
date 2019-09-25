#pragma once
#include <vector>
#include <string>

struct GraphAccessor;
using pos_id = uint64_t;
using vecf =  std::vector<float>;
GraphAccessor * create_graph_accessor(const char * accessor_path,int position_size);
void free_graph_accessor(GraphAccessor * ba);
void add_positions(GraphAccessor * ba, const std::vector<float> & positions, const std::vector<pos_id> &ids_vec);
//endline seperated list of ids
void delete_position(GraphAccessor * ba,pos_id id);
void update_position(GraphAccessor * ba,const float * position,pos_id id);
std::vector<pos_id> fetch_similar(GraphAccessor * ba,const float * position, int fetch_count);
vecf fetch_vec(GraphAccessor * ba,pos_id id);
