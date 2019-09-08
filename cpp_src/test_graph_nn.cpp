#include <iostream>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include "graph_nn.h"
#include"cnpy.h"

using namespace std;

constexpr int POS_SIZE = 300;
constexpr int NUM_POS = 10000;
const pos_id SPEC_POS_IDX = 412;
vector<float> compare_position(){
    vector<float> pos(POS_SIZE,0.5f);
    return pos;
}
vector<float> compare_position2(){
    vector<float> pos(POS_SIZE);
    std::random_device rd;

    std::mt19937 e2(rd());

    std::normal_distribution<float> dist(0.0f, 10.0f);

    for(float & v : pos){
        v = dist(e2);
    }
    return pos;
}
vector<float> generate_positions(){
    vector<float> all_poses(NUM_POS * POS_SIZE);

    std::random_device rd;

    std::mt19937 e2(rd());

    std::normal_distribution<float> dist(0.0f, 10.0f);
    for(float & v : all_poses){
        v = dist(e2);
    }
    auto cmp_pos = compare_position();
    std::copy(cmp_pos.begin(),cmp_pos.end(),all_poses.begin()+SPEC_POS_IDX*POS_SIZE);
    return all_poses;
}
vector<pos_id> gen_ids(){
    int cur_idx = 0;
    vector<pos_id> all_poses(NUM_POS);
    for(pos_id & v : all_poses){
        v = (cur_idx);
        cur_idx += 1;
    }
    return all_poses;
}

int main(){
    GraphAccessor * accessor = create_graph_accessor("temp",POS_SIZE);
    auto posses = generate_positions();

    cnpy::npy_save("arr1.npy",&posses[0],{NUM_POS,POS_SIZE},"w");
    add_all_positions(accessor,{"arr1.npy"});
    auto cmp_pos = compare_position();
    std::vector<pos_id> sim = fetch_similar(accessor,cmp_pos.data(), 0.00001f, 50);
    std::cout << sim.size() << "\n";

    if(sim.size() == 0 || sim.size() > 1 || sim[0] != (SPEC_POS_IDX)){
        std::cout << "failed fetch 1\n";
    }
    free_graph_accessor(accessor);
}
