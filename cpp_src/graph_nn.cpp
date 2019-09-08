#include "graph_nn.h"
struct GraphAccessor{

};
GraphAccessor * create_graph_accessor(const char * accessor_path,int position_size){
    return  new GraphAccessor();
}
void free_graph_accessor(GraphAccessor * ba){

}
void add_all_positions(GraphAccessor * ba,std::vector<std::string> npy_filenames){

}
//endline seperated list of ids
std::vector<pos_id> fetch_similar(GraphAccessor * ba,const float * position, float max_distance, int max_count){
    return std::vector<pos_id>();
}
vecf fetch_vec(GraphAccessor * ba,pos_id id);
