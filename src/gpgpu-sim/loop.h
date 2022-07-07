//
// Created by pouria on 7/1/22.
//

#ifndef ORIGINAL_FREQ_PER_SM_LOOP_H
#define ORIGINAL_FREQ_PER_SM_LOOP_H
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <array>
#include <random>
#include <bitset>
typedef std::bitset<32> mask_t;
struct Node_P{
  unsigned period;
  struct Node_P* left;
  struct Node_P* right;
  unsigned warp_id;
};

class Tree_P{
 public:
  Tree_P();
  struct Node_P* root;
  struct Node_P* max_node;
  unsigned Tree_size;
  void insert_new_loop(unsigned cycles,struct Node_P* root,unsigned warp_id);
  unsigned find_max_period(struct Node_P* node);
};

class CLS{
 public:
  CLS();
    std::vector<std::vector<std::vector<unsigned >>> CLS_TABLE;
    void push(unsigned T,unsigned B,unsigned warp_id,mask_t mask,unsigned cycles);
    int search_T(unsigned T,unsigned warp_id);
    void update(unsigned B,unsigned idx,unsigned warp_id);
    void pop(unsigned idx,unsigned warp_id);
    void print_cls_table(unsigned warp_id,unsigned c_id,unsigned s_id,long long unsigned cycles);
    void flush();
    void end_loop_exe(unsigned pc,unsigned warp_id);
    std::vector<mask_t>Masks;
    void update_cycle(unsigned idx,unsigned cycles,mask_t mask,unsigned warp_id);
    std::vector<unsigned >biggest_loop_info;
    class Tree_P* P_tree;
    unsigned search_biggest_period();

};



//void insert()

#endif  // ORIGINAL_FREQ_PER_SM_LOOP_H
