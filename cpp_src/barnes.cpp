#include "barnes.h"
#include <cstring>
#include <memory>
#include <fstream>
#include <cmath>
#include <cassert>
#include <iostream>
#include <unordered_map>
/*struct SpacialTree{

};*/
constexpr int MAX_LEAF_SIZE = 5;
constexpr pos_id NULL_POS = pos_id(-1);

struct SpacialNode{
    std::unique_ptr<SpacialNode> left;
    std::unique_ptr<SpacialNode> right;
    pos_id leaf_id=NULL_POS;
};

struct BarnesAccessor{
    std::string accessor_path;
    int pos_size;
    pos_id current_pos;
    std::fstream cur_position_file;
    std::unique_ptr<SpacialNode> root;
    std::vector<vecf> pos_datas;
};
struct WalkData{
    int cur_splitdim;
    int num_dim;
    float cur_blocksize;
    vecf block_pos;
};
void walk_down(WalkData & wd,bool went_right){
    float alt_size = wd.cur_blocksize / 2;
    wd.block_pos[wd.cur_splitdim] += went_right * alt_size;

    wd.cur_splitdim += 1;
    if(wd.cur_splitdim >= wd.num_dim){
        wd.cur_blocksize /= 2;
        wd.cur_splitdim = 0;
    }
}
void walk_up(WalkData & wd,bool went_right){
    wd.cur_splitdim -= 1;
    if(wd.cur_splitdim < 0){
        wd.cur_blocksize *= 2;
        wd.cur_splitdim = wd.num_dim - 1;
    }
    float alt_size = wd.cur_blocksize / 2;
    wd.block_pos[wd.cur_splitdim] -= went_right * alt_size;
}
float boundary(WalkData & wd){
    return wd.block_pos[wd.cur_splitdim] + wd.cur_blocksize/2;
}
bool is_split_right(WalkData & wd, const vecf & target){
    float right_pos = boundary(wd);
    float target_pos = target[wd.cur_splitdim];
    return (target_pos > right_pos);
}
/*bool right_in_bound(WalkData & wd, const vecf & target, float maxdist){
    float bound = boundary(wd);
    float target_loc = target[wd.cur_splitdim];
    return target_loc + maxdist >= bound;
}
bool left_in_bound(WalkData & wd, const vecf & target, float maxdist){
    float bound = boundary(wd);
    float target_loc = target[wd.cur_splitdim];
    return target_loc - maxdist <= bound;
}*/
bool is_leaf(SpacialNode & n){
    return !n.left && !n.right;
}
std::unique_ptr<SpacialNode> new_spec(){
    SpacialNode * node = new SpacialNode;
    //node->left = std::unique_ptr<SpacialNode>(new SpacialNode);
    //node->right = std::unique_ptr<SpacialNode>(new SpacialNode);
    return std::unique_ptr<SpacialNode>(node);
}
SpacialNode & walk_to_leaf(WalkData & wd,SpacialNode & cur_node,const vecf & target){
    if(is_leaf(cur_node)){
        return cur_node;
    }
    bool split_right = is_split_right(wd,target);
    SpacialNode * next_node = split_right ? cur_node.right.get() : cur_node.left.get();
    if(next_node == nullptr){
        if(split_right){
            cur_node.right = new_spec();
            return *cur_node.right.get();
        }
        else{
            cur_node.left = new_spec();
            return *cur_node.left.get();
        }
    }
    else{
        walk_down(wd,split_right);
        return walk_to_leaf(wd,*next_node,target);
    }
}
WalkData root_data(int num_dim){
    WalkData root_walk;
    root_walk.cur_splitdim = 0;
    root_walk.block_pos.assign(num_dim,-1.0f);
    root_walk.cur_blocksize = 2;
    root_walk.num_dim = num_dim;
    return root_walk;
}
SpacialNode & get_ref(std::unique_ptr<SpacialNode> & ptr){
    return *ptr.get();
}
void add_node_until_leaf(WalkData & wd, SpacialNode & leaf,
                         pos_id pos1, const vecf & pvec1,
                         pos_id pos2, const vecf & pvec2){
    bool is_right1 = is_split_right(wd,pvec1);
    bool is_right2 = is_split_right(wd,pvec2);
    if(is_right1 == is_right2){
        //std::cout << "uneven\n";
        bool is_right = is_right1;
        SpacialNode * new_node;
        if(is_right){
            leaf.right = new_spec();
            new_node = leaf.right.get();
        }
        else{
            leaf.left =  new_spec();
            new_node = leaf.left.get();
        }
        walk_down(wd,is_right);
        add_node_until_leaf(wd,*new_node,
                            pos1,pvec1,
                            pos2,pvec2);
    }
    else{
        //std::cout << "even\n";
        leaf.left = new_spec();
        leaf.right = new_spec();
        pos_id posl = !is_right1 ? pos1 : pos2;
        pos_id posr = !is_right1 ? pos2 : pos1;
        leaf.left->leaf_id = posl;
        leaf.right->leaf_id = posr;
    }
}
void add_node(BarnesAccessor * ba, const vecf & target, pos_id pos){
    if(!ba->root){
        ba->root = new_spec();
        ba->root->leaf_id = pos;
    }
    else{
        WalkData wd = root_data(target.size());
        SpacialNode & leaf = walk_to_leaf(wd,*ba->root.get(),target);
        if(leaf.leaf_id == NULL_POS){
            //std::cout << "leafed\n";
            leaf.leaf_id = pos;
        }
        else{
            pos_id pid1 = pos;
            const vecf & pvec1 = target;
            pos_id pid2 = leaf.leaf_id;
            const vecf pvec2 = fetch_vec(ba,pid2);

            add_node_until_leaf(wd,leaf,
                                pid1,pvec1,
                                pid2,pvec2);
            leaf.leaf_id = NULL_POS;
        }
    }
}
float sqr(float x){
    return x * x;
}
float distance(const vecf & v1,const vecf & v2){
    assert(v1.size() == v2.size());
    float sum = 0;
    for(int i = 0; i < v1.size(); i++){
        sum += sqr(v1[i] - v2[i]);
    }
    return sqrt(sum);
}
struct DistWalkData{
    vecf dim_dists;
    float tot_dist;
};
bool diff_ok(float diff, float max_diff){
    return sqrt(diff) < max_diff;
}
void fetch_close_ids_rec(BarnesAccessor * ba, WalkData & wd, SpacialNode & cur_node, const vecf & target, std::vector<pos_id> & sim_ids, bool has_diffed, float cur_diff, float max_diff, int max_count){
    if(is_leaf(cur_node)){
        vecf cur_vec = fetch_vec(ba,cur_node.leaf_id);
        if(distance(cur_vec,target) <= max_diff){
            sim_ids.push_back(cur_node.leaf_id);
        }
    }
    bool actual_dir = is_split_right(wd,target);

    float new_block_alt = wd.cur_blocksize/4;
    float old_blockcen = boundary(wd);//.block_pos[wd.cur_splitdim] + wd.cur_blocksize/2;
    float olf_diff_alt = sqr(old_blockcen - target[wd.cur_splitdim]);
    float diff_alt_left = sqr((old_blockcen - new_block_alt) - target[wd.cur_splitdim]);
    float diff_alt_right = sqr((old_blockcen + new_block_alt) - target[wd.cur_splitdim]);
    float new_diff_left = cur_diff + diff_alt_left - olf_diff_alt;
    float new_diff_right = cur_diff + diff_alt_right - olf_diff_alt;

    float blocksize_dimwidth = wd.cur_blocksize/2;//sqrt((wd.num_dim - wd.cur_splitdim)*sqr(wd.cur_blocksize/2) + wd.cur_splitdim*sqr(wd.cur_blocksize/4));
    //std::cout << blocksize_dimwidth << "\n";
    bool left_ok = sqrt(new_diff_left) < blocksize_dimwidth + max_diff;
    bool right_ok = sqrt(new_diff_right) < blocksize_dimwidth + max_diff;
    if(cur_node.left && (left_ok || !has_diffed)){
        bool split_right = false;
        bool left_diffed = has_diffed || split_right != actual_dir;
        walk_down(wd,split_right);
        fetch_close_ids_rec(ba,wd,*cur_node.left.get(),target,sim_ids,left_diffed,new_diff_left,max_diff,max_count);
        walk_up(wd,split_right);
    }
    if(sim_ids.size() > max_count){
        return;
    }
    if(cur_node.right && (right_ok || !has_diffed)){
        //std::cout << "arg\n";
        bool split_right = true;
        bool right_diffed = has_diffed || split_right != actual_dir;
        walk_down(wd,split_right);
        fetch_close_ids_rec(ba,wd,*cur_node.right.get(),target,sim_ids,right_diffed,new_diff_right,max_diff,max_count);
        walk_up(wd,split_right);
    }
}


//store list of positions in seperate files
//store list of ids in single file

//void read_position(pos_id position_id,)
BarnesAccessor * create_barnes_accessor(const char * accessor_path,int position_size){
    BarnesAccessor * ba = new BarnesAccessor;
    ba->accessor_path = std::string(accessor_path);
    ba->pos_size = position_size;
    ba->current_pos = 0;

    std::string pos_filename = "position.vecs";

    ba->cur_position_file.open(ba->accessor_path+pos_filename,std::ios::binary);
    return ba;
}
void free_barnes_accessor(BarnesAccessor * ba){
    delete ba;
}
void normalize(vecf & l){
    size_t size = l.size();
    float sum = 0;
    for(int i = 0; i < size; i++){
        sum += l[i] * l[i];
    }
    float inv = 1.0f/(sqrt(sum)+0.0000001f);
    for(int i = 0; i < size; i++){
        l[i] *= inv;
    }
}
void add_position(BarnesAccessor * ba, pos_id id, const float * position){
    if(id != ba->current_pos){
        exit(10);
    }
    vecf pos_vec(position,position+ba->pos_size);
    normalize(pos_vec);
    add_node(ba,pos_vec,ba->current_pos);
    ba->pos_datas.push_back(pos_vec);
    ba->current_pos++;
}
void update_position(BarnesAccessor * ba, pos_id id, const float * position){
    vecf pos_vec(position,position+ba->pos_size);
    normalize(pos_vec);
    ba->pos_datas.at(id) = pos_vec;
}
//endline seperated list of ids
std::vector<pos_id> fetch_similar(BarnesAccessor * ba,const float * position, float max_distance, int max_count){
    std::vector<pos_id> res;
    if(!ba->root){
        return res;
    }
    vecf pos_vec(position,position+ba->pos_size);
    normalize(pos_vec);
    WalkData root = root_data(ba->pos_size);
    float sqr_diff_cen = distance(pos_vec,vecf(ba->pos_size,0.0f));
    float start_diff = sqr_diff_cen;
    fetch_close_ids_rec(ba,root,*ba->root.get(),pos_vec,res,false,start_diff,max_distance,max_count);
    return res;
}
vecf fetch_vec(BarnesAccessor * ba,pos_id id){
    return ba->pos_datas.at(id);
}
