#include "graph_nn.h"
#include "cnpy.h"
#include "data_accessor.h"
#include <cassert>
#include <algorithm>

constexpr uint64_t MAX_ARRAY_SIZE = 2*(1LL<<30);
using idx_ty = uint64_t;
using pos_ty = uint64_t;

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
template<class item_ty>
class ValDataAccessor{
private:
    DataAccessor accessor;
public:
    ValDataAccessor(std::string folder):
        accessor(folder,sizeof(item_ty)){
    }
    size_t num_items()const{
        return accessor.num_items();
    }
    void add_data(const item_ty & item){
        accessor.add_data((char*)(&item),1);
    }
    void add_data(const std::vector<item_ty> & items){
        size_t add_count = items.size();
        accessor.add_data((char*)items.data(),add_count);
    }
    item_ty get_item(size_t pos)const{
        item_ty out_data;
        accessor.get_item((char*)(&out_data),pos);
        return out_data;
    }
};
using FVecDataAccessor = VecDataAccessor<float>;
using FixedGraphDataAccessor = VecDataAccessor<pos_ty>;
struct IdPair{
    pos_id id;
    pos_ty pos;
};
using IdAccessor = ValDataAccessor<IdPair>;

struct Layer{
    FixedGraphDataAccessor layer_accessor;
};
struct GraphAccessor{
    std::string path;
    size_t num_dim;
    std::vector<Layer> layers;
    FVecDataAccessor vec_datas;
    IdAccessor all_ids;
    GraphAccessor(std::string folder,size_t in_num_dim):
        path(folder),
        num_dim(in_num_dim),
        vec_datas(folder,num_dim),
        all_ids(folder){

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
void add_all_positions(GraphAccessor * ba,std::vector<std::pair<std::string,std::string>> npy_filenames){
    for(auto namepair : npy_filenames){
        cnpy::NpyArray data = cnpy::npy_load(namepair.first);
        cnpy::NpyArray ids = cnpy::npy_load(namepair.second);

        vecf all_data = data.as_vec<float>();
        std::vector<pos_ty> ids_vec = ids.as_vec<uint64_t>();

        std::vector<IdPair> id_pairs(ids_vec.size());
        pos_ty start_pos = ba->vec_datas.num_items();
        for(size_t i = 0; i < id_pairs.size(); i++){
            id_pairs[i] = IdPair{ids_vec[i],i+start_pos};
        }

        ba->vec_datas.add_data(all_data);
        ba->all_ids.add_data(id_pairs);
    }
}
//endline seperated list of ids
std::vector<pos_id> fetch_similar(GraphAccessor * ba,const float * position, float max_distance, int max_count){
    return std::vector<pos_id>();
}
template <class RandIterator, class T,class CmpFnTy>
  RandIterator my_lower_bound (RandIterator first, RandIterator last, const T& val,CmpFnTy cmp_fn)
{
  RandIterator it;
  RandIterator count, step;
  count = last - first;
  while (count>0)
  {
    it = first; step=count/2; it += step;
    if (cmp_fn(it,val)) {                 // or: if (comp(*it,val)), for version (2)
      first=++it;
      count-=step+1;
    }
    else count=step;
  }
  return first;
}
pos_ty get_pos(IdAccessor & accessor,pos_id id){
    pos_id begin = 0;
    pos_id end = accessor.num_items();
    pos_id lower_bound_idx = my_lower_bound(begin,end,id,[&](pos_id idx1,pos_id id){
        return accessor.get_item(idx1).id < id;
    });
    IdPair out_item = accessor.get_item(lower_bound_idx);
    assert(out_item.id == id);
    return out_item.pos;
}
vecf fetch_vec(GraphAccessor * ba,pos_id id){
    pos_ty pos = get_pos(ba->all_ids,id);
    vecf res(ba->num_dim);
    ba->vec_datas.get_item(res,pos);
    return res;
}
