#include <iostream>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include <ctime>
#include <map>
#include <unordered_set>
#include "graph_nn.h"
#include"cnpy.h"

using namespace std;

constexpr int POS_SIZE = 300;
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
vector<float> generate_positions(int NUM_POS){
    vector<float> all_poses(NUM_POS * POS_SIZE);

    std::random_device rd;

    std::mt19937 e2(rd());

    std::normal_distribution<float> dist(0.0f, 10.0f);
    for(float & v : all_poses){
        v = dist(e2);
    }
    return all_poses;
}
vector<pos_id> gen_ids(int NUM_POS,int start_id=0){
    int cur_idx = 5+start_id;
    vector<pos_id> all_poses(NUM_POS);
    for(pos_id & v : all_poses){
        v = (cur_idx);
        cur_idx += 2;
    }
    return all_poses;
}
void simple_test(){
    constexpr int NUM_POS = 10000;
    const pos_id SPEC_POS_IDX = 412;

    GraphAccessor * accessor = create_graph_accessor("temp",POS_SIZE);
    auto posses = generate_positions(NUM_POS);
    auto cmp_pos = compare_position();
    std::copy(cmp_pos.begin(),cmp_pos.end(),posses.begin()+SPEC_POS_IDX*POS_SIZE);
    vector<pos_id> ids = gen_ids(NUM_POS);

    add_positions(accessor,posses,ids);
    std::vector<pos_id> sim = fetch_similar(accessor,cmp_pos.data(), 1);
    std::cout << sim.size() << "\n";
    if(sim.size() == 0 || sim.size() > 1 || sim[0] != (SPEC_POS_IDX)){
       std::cout << "failed fetch 1\n";
    }
    vecf out_data = fetch_vec(accessor,ids[SPEC_POS_IDX]);
    std::cout << out_data[0] << "  " << out_data[1] << "\n";
    free_graph_accessor(accessor);
}
const int performance_arr_count = 1;
const int POS_PER_PERF_ARR = 100000;
void build_performance_test(){
    pos_id cur_id = 0;
    for(int arridx = 0; arridx < performance_arr_count; arridx++){
        auto posses = generate_positions(POS_PER_PERF_ARR);
        vector<pos_id> ids = gen_ids(POS_PER_PERF_ARR,cur_id);
        cur_id = ids.back();

        string arr_name = "arr" + to_string(arridx)+".npy";
        string ids_name = "ids" + to_string(arridx)+".npy";
        cnpy::npy_save(arr_name,&posses[0],{POS_PER_PERF_ARR,POS_SIZE},"w");
        cnpy::npy_save(ids_name,&ids[0],{POS_PER_PERF_ARR},"w");
    }
}
void add_all_positions(GraphAccessor * ba,std::vector<std::pair<std::string,std::string>> npy_filenames){
    for(auto namepair : npy_filenames){
        cnpy::NpyArray data = cnpy::npy_load(namepair.first);
        cnpy::NpyArray ids = cnpy::npy_load(namepair.second);

        vecf all_data = data.as_vec<float>();
        std::vector<pos_id> ids_vec = ids.as_vec<uint64_t>();
        assert(ids.shape[0] == data.shape[0]);

        add_positions(ba,all_data,ids_vec);
    }
}
void performance_test(){
    cout << "started adding positions" << endl;
    GraphAccessor * accessor = create_graph_accessor("temp",POS_SIZE);
    vector<pair<string,string>> files;
    for(int arridx = 0; arridx < performance_arr_count; arridx++){
        string arr_name = "arr" + to_string(arridx)+".npy";
        string ids_name = "ids" + to_string(arridx)+".npy";
        add_all_positions(accessor,{make_pair(arr_name,ids_name)});
    }
    int NUM_QUERRIES = 100000;
    int num_iters = 10;
    cout << "finished adding positions" << endl;
    auto posses = generate_positions(NUM_QUERRIES);
    cout << "started querrying" << endl;
    for(int i = 0; i < num_iters; i++){
        int start = clock();
        for(int j = 0; j < NUM_QUERRIES; j++){
            fetch_similar(accessor,posses.data()+j*POS_SIZE,5);
        }
        int end = clock();
        cout << "time for iter " << i << ": " << (end - start)/float(CLOCKS_PER_SEC) << endl;
    }
    free_graph_accessor(accessor);
}
float dot(float * d1,float * d2,size_t size){
    float sum = 0;
    for(size_t i = 0; i < size; i++){
        sum += d1[i] * d2[i];
    }
    return sum;
}
vector<vector<pos_id>> ideal_ranking(vecf posses, vecf querries,int rank_size){
    size_t num_querries = querries.size()/POS_SIZE;
    size_t num_posses = posses.size()/POS_SIZE;
    vector<vector<pos_id>> closest_list(num_querries);
    for(pos_id i = 0; i < num_querries; i++){
        map<float,pos_id> pos_vals;
        for(pos_id j = 0; j < num_posses; j++){
            float distance = dot(&posses[j*POS_SIZE],&querries[i*POS_SIZE],POS_SIZE);
            if(pos_vals.size() < rank_size){
                pos_vals[distance] = j;
            }
            else if(pos_vals.begin()->second < distance){
                pos_vals.erase(pos_vals.begin());
                pos_vals[distance] = j;
            }
        }
        for(auto pos_pair : pos_vals){
            closest_list[i].push_back(pos_pair.second);
        }
    }
    return  closest_list;
}
void ranking_test(){
    const size_t NUM_POS = 100;
    const size_t NUM_QUERRIES = 200;
    cout << "arg" << endl;
    auto posses = generate_positions(NUM_POS);
    auto querries = generate_positions(NUM_QUERRIES);
    vector<pos_id> ids(NUM_POS);
    cout <<posses.size() << endl;
    for(pos_id i = 0; i < NUM_POS; i++){
        ids[i] = i;
    }
    size_t num_to_rank = 5;
    vector<vector<pos_id>> true_ranking = ideal_ranking(posses,querries,num_to_rank);
    GraphAccessor * accessor = create_graph_accessor("temp",POS_SIZE);
    add_positions(accessor,posses,ids);
    int diff_count = 0;
    for(size_t i = 0; i < NUM_QUERRIES; i++){
        std::vector<pos_id> algo_ranking = fetch_similar(accessor,&querries[i*POS_SIZE],num_to_rank);

        //uniqueness test
        unordered_set<pos_id> true_r(true_ranking[i].begin(),true_ranking[i].end());
        unordered_set<pos_id> algo_r(algo_ranking.begin(),algo_ranking.end());
        assert(algo_r.size() == true_r.size());
        assert(algo_r.size() == num_to_rank);

        //correctness_test
        for(pos_id al_r : algo_ranking){
            if(!true_r.count(al_r)){
                diff_count++;
            }
        }
    }
    cout << "found " << diff_count << " diffs out of " << NUM_QUERRIES*num_to_rank << " possiblities\n";
}

int main(){
    //ranking_test();
    build_performance_test();
    //simple_test();
}
