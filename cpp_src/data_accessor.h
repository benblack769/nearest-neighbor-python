#pragma once
#undef NDEBUG
#include <vector>
#include <fstream>

class AccessCacher{

};

class DataAccessor{
private:
    //std::fstream working_file;
    size_t type_size;
    std::vector<char> raw_data;
    size_t stream_pos;
public:
    DataAccessor(std::string folder,size_t in_type_size){
        type_size = in_type_size;
        stream_pos = 0;
    }
    size_t num_items()const{
        return raw_data.size() / type_size;
    }
    void add_data(const char * c_data,size_t num_items){
        raw_data.insert(raw_data.end(), c_data, c_data + num_items*type_size);
    }
    void edit_item(size_t pos, const char * c_data){
        size_t start = type_size*pos;
        std::copy(c_data,c_data+type_size,raw_data.begin()+start);
    }
    void get_item(char * out_data,size_t pos)const{
        size_t item_idx = pos * type_size;
        std::copy(raw_data.begin()+item_idx,raw_data.begin()+item_idx+type_size,out_data);
    }
    void reset_stream(){
        stream_pos = 0;
    }
    bool stream_ended(){
        return stream_pos*type_size >= raw_data.size();
    }
    bool get_next_data(char * out_data){
        if(stream_ended()){
            return false;
        }
        size_t stream_idx = stream_pos * type_size;
        std::copy(raw_data.begin()+stream_idx,raw_data.begin()+stream_idx+type_size,out_data);
        stream_pos++;
        return true;
    }
};
