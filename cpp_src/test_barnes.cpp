#include <iostream>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include "barnes.h"

using namespace std;

constexpr int POS_SIZE = 300;
constexpr int NUM_POS = 10000;
const pos_id SPEC_POS_IDX = 412;
vector<float> compare_position(){
    vector<float> pos(POS_SIZE,0.5f);
    return pos;
}
vector<float> compare_position2(){
    vector<float> pos(POS_SIZE,0.1f);
    pos[4] = -1;
    return pos;
}
vector<float> generate_positions(){
    vector<float> all_poses(NUM_POS * POS_SIZE);

    std::random_device rd;

    std::mt19937 e2(rd());

    std::normal_distribution<float> dist(0, 10);
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
    BarnesAccessor * accessor = create_barnes_accessor("temp",POS_SIZE);
    auto posses = generate_positions();
    auto ids = gen_ids();
    for(int i = 0; i < NUM_POS; i++){
        add_position(accessor,ids[i],&posses[i*POS_SIZE]);
    }
    auto cmp_pos = compare_position();
    std::vector<pos_id> sim = fetch_similar(accessor,cmp_pos.data(), 0.000001f, 50);
    std::cout << sim.size() << "\n";
    //std::cout << sim[0] << "\n";

    if(sim.size() == 0 || sim.size() > 1 || sim[0] != (SPEC_POS_IDX)){
        std::cout << "failed fetch 1\n";
    }
    auto cmp_pos2 = compare_position2();
    update_position(accessor, (SPEC_POS_IDX),cmp_pos2.data());
    std::vector<pos_id> sim2 = fetch_similar(accessor,cmp_pos2.data(), 0.3, 10000);
    std::cout << sim2.size() << "\n";
    vecf outvec = fetch_vec(accessor,sim2[3]);
    for(float v : outvec){
        cout << v << " ";
    }
    cout << "\n";
    if(sim2.size() == 0 || sim2.size() > 1 || sim2[0] != (SPEC_POS_IDX)){
        std::cout << "failed fetch 2\n";
    }
    free_barnes_accessor(accessor);
}
