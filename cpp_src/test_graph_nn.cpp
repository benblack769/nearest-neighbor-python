#include <iostream>
#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include <ctime>
#include <map>
#include <unordered_set>
#include "graph_nn.h"
#include "cnpy.h"
#include "local_ranker.h"
#include "fast_dot_prod.h"

using namespace std;

constexpr int POS_SIZE = 300;
vector<float> compare_position(){
    vector<float> pos(POS_SIZE,2.0f);
    return pos;
}
size_t rand_count = 0;
vector<float> generate_positions(int NUM_POS){
    vector<float> all_poses(NUM_POS * POS_SIZE);

    //std::random_device rd;
    std::mt19937 e2(rand_count);
    rand_count+=1;

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
int test_fail_count = 0;
int test_count = 0;
void test_assert(bool cond,string name){
    test_count++;
    if(!cond){
        cout << name << ": FAILED" << endl;
        test_fail_count++;
    }
    else{
        cout << name << ": PASSED" << endl;
    }
}
void id_test(){
    GraphAccessor * accessor = create_graph_accessor("temp",POS_SIZE);
    auto posses1 = generate_positions(10);
    auto posses2 = generate_positions(10);
    auto posses3 = generate_positions(10);
    auto ids1 = gen_ids(10,100);
    auto ids2 = gen_ids(10,1);
    auto ids3 = gen_ids(10,10);
    add_positions(accessor,posses1,ids1);
    add_positions(accessor,posses2,ids2);
    add_positions(accessor,posses3,ids3);
    vecf res22 = fetch_vec(accessor,ids2[0]);
    test_assert(std::equal(res22.begin(),res22.end(),posses2.begin()),"id_test");

    free_graph_accessor(accessor);
}
void simple_test(){
    constexpr int NUM_POS = 1000;
    const pos_id SPEC_POS_IDX = 412;

    GraphAccessor * accessor = create_graph_accessor("temp",POS_SIZE);
    auto posses = generate_positions(NUM_POS);
    auto cmp_pos = compare_position();
    std::copy(cmp_pos.begin(),cmp_pos.end(),posses.begin()+SPEC_POS_IDX*POS_SIZE);
    vector<pos_id> ids = gen_ids(NUM_POS);

    add_positions(accessor,posses,ids);
    std::vector<ValIdPair> sim1 = fetch_similar(accessor,cmp_pos.data(), 1, 1000000);
    std::vector<ValIdPair> sim2 = fetch_similar(accessor,cmp_pos.data(), 1, 1000000);
    std::vector<ValIdPair> sim3 = fetch_similar(accessor,cmp_pos.data(), 1, 1000000);
    test_assert((sim1.size() == 1 && sim1[0].id == ids.at(SPEC_POS_IDX)) ||
                (sim2.size() == 1 && sim2[0].id == ids.at(SPEC_POS_IDX)) ||
                (sim3.size() == 1 && sim3[0].id == ids.at(SPEC_POS_IDX))
            ,"simplefetch1");
    auto new_cmp_pos = cmp_pos;
    for(float & v : new_cmp_pos){
        v = -4;
    }
    update_position(accessor,new_cmp_pos.data(),SPEC_POS_IDX);
    std::vector<ValIdPair> sim = fetch_similar(accessor,cmp_pos.data(), 1, 1000000);
    test_assert(sim.size() == 1 && sim[0].id != ids.at(SPEC_POS_IDX),"updatedfetch1");
    sim = fetch_similar(accessor,new_cmp_pos.data(), 1, 1000000);
    test_assert(sim.size() == 1 && sim[0].id == ids.at(SPEC_POS_IDX),"simplefetch2");

    //vecf out_data = fetch_vec(accessor,ids[SPEC_POS_IDX]);
    //std::cout << out_data[0] << "  " << out_data[1] << "\n";
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
            fetch_similar(accessor,posses.data()+j*POS_SIZE,5, 1000000);
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
vector<vector<ValIdPair>> ideal_ranking(vecf posses, vecf querries,int rank_size){
    size_t num_querries = querries.size()/POS_SIZE;
    size_t num_posses = posses.size()/POS_SIZE;
    vector<vector<ValIdPair>> closest_list(num_querries);
    for(pos_id i = 0; i < num_querries; i++){
        map<float,pos_id> pos_vals;
        for(pos_id j = 0; j < num_posses; j++){
            float distance = fast_dot_prod(&posses[j*POS_SIZE],&querries[i*POS_SIZE],POS_SIZE);
            if(pos_vals.size() < rank_size){
                pos_vals[distance] = j;
            }
            else if(pos_vals.begin()->second < distance){
                pos_vals.erase(pos_vals.begin());
                pos_vals[distance] = j;
            }
        }
        for(auto pos_pair : pos_vals){
            closest_list[i].push_back(ValIdPair{pos_pair.second,pos_pair.first});
        }
    }
    return  closest_list;
}
unordered_set<pos_id> from_idpairs(std::vector<ValIdPair> pairs){
    unordered_set<pos_id> res;
    for(auto p : pairs){
        res.insert(p.id);
    }
    return res;
}
double sqr(double x){
    return x * x;
}
double sim_metric(double val1,double val2){
    return val1 - val2;
}
void ranking_test(){
    const size_t NUM_POS = 1000;
    const size_t NUM_QUERRIES = 2000;
    cout << "arg" << endl;
    auto posses = generate_positions(NUM_POS);
    auto querries = generate_positions(NUM_QUERRIES);
    vector<pos_id> ids(NUM_POS);
    cout <<posses.size() << endl;
    for(pos_id i = 0; i < NUM_POS; i++){
        ids[i] = i;
    }
    size_t num_to_rank = 5;
    vector<vector<ValIdPair>> true_ranking = ideal_ranking(posses,querries,num_to_rank);
    GraphAccessor * accessor = create_graph_accessor("temp",POS_SIZE);
    add_positions(accessor,posses,ids);
    int diff_count = 0;
    double false_diff_sum = 0;
    double true_diff_sum = 0;
    for(size_t i = 0; i < NUM_QUERRIES; i++){
        //fetch_similar(accessor,&querries[i*POS_SIZE],num_to_rank, 100);
    }
    for(size_t i = 0; i < NUM_QUERRIES; i++){
        std::vector<ValIdPair> algo_ranking = fetch_similar(accessor,&querries[i*POS_SIZE],num_to_rank, 100);


        //uniqueness test
        unordered_set<pos_id> true_r = from_idpairs(true_ranking[i]);//.begin(),true_ranking[i].end());
        unordered_set<pos_id> algo_r = from_idpairs(algo_ranking);//(algo_ranking.begin(),algo_ranking.end());
        assert(algo_r.size() == num_to_rank && "sizetest2");
        assert(algo_r.size() == true_r.size());
        assert(true_ranking[i].size() == num_to_rank);

        //correctness_test
        for(ValIdPair al_r : algo_ranking){
            if(!true_r.count(al_r.id)){
                diff_count++;
            }
        }
        for(ValIdPair v : algo_ranking){
            false_diff_sum += v.val;
        }
        for(ValIdPair v : true_ranking[i]){
            true_diff_sum += v.val;
        }
    }
    cout << "found " << diff_count << " diffs out of " << NUM_QUERRIES*num_to_rank << " possiblities\n";
    cout << "true diff: " << true_diff_sum << ". False diff: " << false_diff_sum << " \n";
}
int sigbit_test(){
    test_assert(sigbit_index(0xf12) == 11,"assert12");
    test_assert(sigbit_index(0x212) == 9,"assert10");
    test_assert(sigbit_index(0x212000) == 9+12,"assert21");
    test_assert(sigbit_index(0x2120fff00) == 9+24,"assert21");
    test_assert(sigbit_index(0x1120ff0abcf00) == 12*4,"assert92416");
    test_assert(sigbit_index(0x4120ff0abcf00) == 2+12*4,"assert924162");
}
void time_fast_dot_prod(){
    vecf vec1 = compare_position();
    vecf many_vecs = generate_positions(5000);
    int start = clock();
    float sum = 0;
    for(int j = 0; j < 1000; j++){
        for(int i = 0; i < 5000; i++){
            sum += fast_dot_prod(&vec1[0],&many_vecs[i*POS_SIZE],POS_SIZE);
        }
    }
    int end = clock();
    if(sum != 0){
        cout << "fast dot prod time: " << (end - start)/float(CLOCKS_PER_SEC) << endl;
    }
}
void test_fast_dot_prod(){
    vecf vec1 = generate_positions(1);
    vecf vec2 = generate_positions(1);
    float true_dot = dot(vec1.data(),vec2.data(),POS_SIZE);
    float compare_dot = fast_dot_prod(vec1.data(),vec2.data(),POS_SIZE);
    bool worked = abs(true_dot - compare_dot) < 0.00001 ||
            (true_dot * 0.9999 < compare_dot &&
             compare_dot * 0.9999 < true_dot);
    test_assert(worked,"fast_dot_prod_check");
    cout << true_dot << endl;
    cout << compare_dot << endl;
}
int main(){
    sigbit_test();
    simple_test();
    ranking_test();
    test_fast_dot_prod();
    id_test();
    time_fast_dot_prod();
    //performance_test();
    //simple_test();
    if(test_fail_count == 0){
        cout << test_count << " tests passed\n";
    }
    else{
        cout << test_count << " tests run, " << test_fail_count << " tests failed\n";
    }
}
