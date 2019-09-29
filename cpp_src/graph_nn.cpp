#include "graph_nn.h"
#include "data_accessor.h"
#include <cassert>
#include <algorithm>
#include <map>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include "global_ranker.h"
#include "local_ranker.h"
#include "fast_dot_prod.h"

constexpr uint64_t MAX_ARRAY_SIZE = 2*(1LL<<30);
using idx_ty = uint64_t;
using pos_ty = uint64_t;
using pos_vector = std::vector<pos_ty>;

template<class item_ty>
class VecDataAccessor{
private:
    size_t vec_size;
    DataAccessor accessor;
public:
    using vecty = std::vector<item_ty>;
    VecDataAccessor(std::string folder,size_t in_vec_size):
        vec_size(in_vec_size),
        accessor(folder,in_vec_size*sizeof(item_ty)){
    }
    size_t num_items()const{
        return accessor.num_items();
    }
    void add_data(const vecty & data_buffer){
        assert(data_buffer.size() % vec_size == 0);
        size_t add_count = data_buffer.size()/vec_size;
        accessor.add_data((char*)data_buffer.data(),add_count);
    }
    void edit_item(size_t pos, const vecty & f_data){
        assert(f_data.size() == vec_size);
        accessor.edit_item(pos,(char*)f_data.data());
    }
    void get_item(vecty &  out_data,size_t pos)const{
        assert(out_data.size() == vec_size);
        accessor.get_item((char*)out_data.data(),pos);
    }
    void reset_stream(){
        accessor.reset_stream();
    }
    bool stream_ended(){
        return accessor.stream_ended();
    }
    bool get_next_data(vecty & out_data){
        assert(out_data.size() == vec_size);
        return accessor.get_next_data((char*)out_data.data());
    }
};
using FVecDataAccessor = VecDataAccessor<float>;
struct IdPair{
    pos_id id;
    pos_ty pos;
    bool operator < (const IdPair & other)const{
        return id < other.id;
    }
};
using IdData = std::vector<IdPair>;

struct GraphAccessor{
    std::string path;
    size_t num_dim;
    FVecDataAccessor vec_datas;
    std::unordered_map<pos_id,pos_ty> id_mapping;
    std::vector<pos_id> pos_ids;
    GlobalRanker global_rank;
    LocalRanker local_ranks;
    GraphAccessor(std::string folder,size_t in_num_dim):
        path(folder),
        num_dim(in_num_dim),
        vec_datas(folder,num_dim){

    }
};
size_t first_layer_size(size_t num_dim){
    return num_dim * 2;
}
size_t layer_growth_strategy(size_t old_layer){
    return old_layer * 4;
}
GraphAccessor * create_graph_accessor(const char * accessor_path,int position_size){
    return new GraphAccessor(std::string(accessor_path),position_size);
}
void free_graph_accessor(GraphAccessor * ba){
    delete ba;
}
void add_positions(GraphAccessor * ba,const std::vector<float> & positions, const std::vector<pos_id> & ids_vec){
    size_t num_add = ids_vec.size();
    assert(num_add == positions.size() / ba->num_dim);

    //add vectors
    ba->vec_datas.add_data(positions);

    //add ids
    pos_ty start_pos = ba->pos_ids.size();
    for(size_t i = 0; i < ids_vec.size(); i++){
        assert(ba->id_mapping.count(ids_vec[i]) == 0);
        ba->id_mapping[ids_vec[i]] = i+start_pos;
        ba->pos_ids.push_back(ids_vec[i]);
    }

    //add to global rankings
    for(size_t i = 0; i < num_add; i++){
        ba->global_rank.append_value(5);
        ba->local_ranks.add_node();
    }
}
float dot_prod(const vecf & buf1,const vecf & buf2){
    return fast_dot_prod(buf1.data(),buf2.data(),buf1.size());
}
float sqr(float x){
    return x * x;
}
float pow4(float x){
    return sqr(sqr(x));
}

class QuerryData{
private:
    using loc_ty = uint64_t;
    struct ValLocPair{
        loc_ty loc;
        float val;
    };
    static constexpr loc_ty NO_PARENT = size_t(-1);
    GlobalRanker collected_ranking;
    std::vector<loc_ty> parents;
    std::vector<pos_ty> idxs;
    std::unordered_set<pos_ty> index_set;
    UnifFloatSampler sampler;
    std::vector<float> values;
    vecf vec_buffer;
private:
    std::vector<ValLocPair> calc_ranking(){
        std::vector<ValLocPair> res(idxs.size());
        for(size_t i = 0; i < res.size(); i++){
            res[i] = ValLocPair{.loc=i,.val=values[i]};
        }
        std::sort(res.begin(),res.end(),[](const ValLocPair & v1,const ValLocPair & v2){
            return v1.val > v2.val;
        });
        return res;
    }
    void update_similar(GraphAccessor * ba,const std::vector<ValLocPair> & ranking){
        const int max_similar = 4;
        const int similar_count = std::min(size_t(max_similar),idxs.size());
        std::vector<ValLocPair> similar(ranking.begin(),ranking.begin()+similar_count);

        //update global rankings
        for(ValLocPair source : similar){
            loc_ty loc = source.loc;
            while(loc != NO_PARENT){
                ba->global_rank.add_weight(idxs.at(loc),1);
                loc = parents.at(loc);
            }
        }
        //update local rankings of similars pairwise
        for(ValLocPair source : similar){
            for(ValLocPair dest : similar){
                ba->local_ranks.inc_weight(idxs.at(source.loc),idxs.at(dest.loc));
            }
        }
        //update local rankings to parents
        for(ValLocPair source : similar){
            loc_ty loc = source.loc;
            loc_ty parent = parents.at(source.loc);
            while(parent != NO_PARENT){
                ba->local_ranks.inc_weight(idxs.at(loc),idxs.at(parent));
                ba->local_ranks.inc_weight(idxs.at(parent),idxs.at(loc));
                loc = parent;
                parent = parents.at(loc);
            }
        }
    }
    std::vector<ValIdPair> get_similar(GraphAccessor * ba,const std::vector<ValLocPair> & ranking,int fetch_count){
        std::vector<ValIdPair> best_ids(fetch_count);
        for(int i = 0; i < fetch_count; i++){
            best_ids[i] = ValIdPair{.id=ba->pos_ids.at(idxs.at(ranking.at(i).loc)),
                                    .val=ranking.at(i).val};
        }
        return best_ids;
    }
    pos_ty sample_global(GraphAccessor * ba){
        pos_ty val = ba->global_rank.sample();
        while(index_set.count(val)){
            val = ba->global_rank.sample();
        }
        return val;
    }
    pos_ty sample_local(GraphAccessor * ba,loc_ty parent_loc){
        pos_ty parent = idxs.at(parent_loc);
        if(!ba->local_ranks.can_sample(parent)){
            return sample_global(ba);
        }
        pos_ty child = ba->local_ranks.sample_local(parent);
        if(index_set.count(child)){
            return sample_global(ba);
        }
        return child;
    }
    void init_vb_if_not(GraphAccessor * ba){
        if(vec_buffer.size() != ba->num_dim){
            vec_buffer.resize(ba->num_dim);
        }
    }
    ValPosPair querry(GraphAccessor * ba,const vecf & vec){
        pos_ty new_pos;
        loc_ty parent_loc;
        if(sampler.sample() < 0.3 || !collected_ranking.can_sample()){
            new_pos = sample_global(ba);
            parent_loc = NO_PARENT;
        }
        else{
            parent_loc = collected_ranking.sample();
            new_pos = sample_local(ba,parent_loc);
        }
        size_t new_loc = idxs.size();
        parents.push_back(parent_loc);
        idxs.push_back(new_pos);
        index_set.insert(new_pos);
        collected_ranking.append_value(1);

        init_vb_if_not(ba);
        ba->vec_datas.get_item(vec_buffer,new_pos);
        float dot_prod_val = dot_prod(vec,vec_buffer);

        values.push_back(dot_prod_val);
    }
    void querry_until_limit(GraphAccessor * ba,const vecf & vec,int fetch_count, int max_query){
        for(int i = 0; i < max_query && idxs.size() < ba->global_rank.size()/2; i++){
            querry(ba,vec);
        }
    }
public:
    std::vector<ValIdPair> calc(GraphAccessor * ba,const vecf & vec,int fetch_count, int max_query){
        querry_until_limit(ba,vec,fetch_count,max_query);
        std::vector<ValLocPair> ranking = calc_ranking();
        update_similar(ba,ranking);
        return get_similar(ba,ranking,fetch_count);
    }
};
void normalize(vecf & vec){
    float sum = 0;
    for(float v : vec){
        sum += v;
    }
    float inv_sum = 1.0f/sum;
    for(float & v : vec){
        v *= inv_sum;
    }
}
//endline seperated list of ids
std::vector<ValIdPair> fetch_similar(GraphAccessor * ba, const float * position, int fetch_count,int max_querry){
    vecf pos_vec(position,position+ba->num_dim);
    QuerryData querry;
    return querry.calc(ba,pos_vec,fetch_count,max_querry);
}
void update_position(GraphAccessor * ba,const float * position,pos_id id){
    //throw "not implemented";
}
vecf fetch_vec(GraphAccessor * ba,pos_id id){
    pos_ty pos = ba->id_mapping.at(id);
    std::cout << pos << "\n";
    vecf res(ba->num_dim);
    ba->vec_datas.get_item(res,pos);
    return res;
}
