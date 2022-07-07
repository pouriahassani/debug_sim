//
// Created by pouria on 7/1/22.
//
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <array>
#include <random>
#include "loop.h"

CLS::CLS() {
  CLS_TABLE.resize(48);
  for(int i=0;i<48;i++)
    Masks.push_back(0);
  P_tree = new Tree_P();
}

void CLS::push(unsigned int T, unsigned int B,unsigned warp_id,mask_t mask,unsigned cycles) {
  std::vector<unsigned >V;
  V.push_back(T);
  V.push_back(B);
  for(int i=0;i<32;i++){
      V.push_back(mask[i]);
      V.push_back(cycles);
      V.push_back(0);
  }
  CLS_TABLE[warp_id].push_back(V);
  Masks[warp_id] |= mask;

}

int CLS::search_T(unsigned int T,unsigned warp_id) {
  int counter = 0;
  std::vector<std::vector<unsigned>>::iterator it;
  for(it = CLS_TABLE[warp_id].begin();it != CLS_TABLE[warp_id].end();++it){
    if((*it)[0] == T)
      return counter;
    counter ++;
  }
  return -1;
}

void CLS::update(unsigned B,unsigned idx,unsigned warp_id){
  CLS_TABLE[warp_id][idx][1] = B;
}

void CLS::pop(unsigned idx,unsigned warp_id){
  CLS_TABLE[warp_id].erase(CLS_TABLE[warp_id].begin() + idx,CLS_TABLE[warp_id].end());
}

void CLS::print_cls_table(unsigned warp_id,unsigned c_idx,unsigned s_idx,long long unsigned cycles) {
  std::ofstream file;
  file.open("/home/pouria/Desktop/G_GPU/DATA/cluster_turn" + std::to_string(c_idx), std::ios::app);
  file<<std::endl;
  file<<cycles<<"\t"<<CLS_TABLE[warp_id].size()<<std::endl;
  for(int i = 0;i<CLS_TABLE[warp_id].size();i++){
    file<<warp_id<<"\t"<<CLS_TABLE[warp_id][i][0]<<"\t"<<CLS_TABLE[warp_id][i][1]<<std::endl;
    for(int j=0;j<32;j++)
    file<<cycles<<"\t"<<CLS_TABLE[warp_id][i][j*3+2]<<"\t"<<CLS_TABLE[warp_id][i][j*3+1+2]<<"\t"<<CLS_TABLE[warp_id][i][j*3+2+2]<<std::endl;
  }
  file.close();
}

void CLS::end_loop_exe(unsigned pc,unsigned warp_id){
  for(unsigned i=0;i<CLS_TABLE[warp_id].size();i++){
    if(pc<CLS_TABLE[warp_id][i][0] || pc>CLS_TABLE[warp_id][i][1]) {


      CLS_TABLE[warp_id].erase(CLS_TABLE[warp_id].begin() + i);
    }
  }

}

void CLS::update_cycle(unsigned idx,unsigned cycles,mask_t mask,unsigned warp_id){
    for(int i=0;i<32;i++){
      if(mask[i]) {
        CLS_TABLE[warp_id][idx][2 + i * 3] = 1;
        CLS_TABLE[warp_id][idx][2 + i * 3 + 2] = cycles - CLS_TABLE[warp_id][idx][2 + i * 3 + 1];
        CLS_TABLE[warp_id][idx][2 + i * 3 + 1] = cycles;
      }
    }
}

Tree_P::Tree_P(){
  root = NULL;
  max_node = NULL;
  Tree_size = 0;
}

//void Tree_P::insert_new_loop(unsigned int cycles,struct Node_P* node,unsigned warp_id) {
//  if(node == NULL){
//    node = (struct Node_P*)malloc(sizeof(struct Node_P));
//    node->right = NULL;
//    node->left = NULL;
//    node->period = cycles;
//    node->warp_id = warp_id;
//    Tree_size++;
//  }
//  else{
//    if(cycles < node->period){
//      insert_new_loop(cycles,node->left);
//    }
//    if(cycles > node->period){
//      insert_new_loop(cycles,node->right);
//    }
//  }
//
//}

unsigned Tree_P::find_max_period(struct Node_P* node){
  if(node->right == NULL)
    return node->period;
  else
    find_max_period(node->right);
}

//void Tree_P::update_Tree_P()
unsigned CLS::search_biggest_period() {
  unsigned max = 0;
  for(int i=0;i<CLS_TABLE.size();i++){
    if(CLS_TABLE[i].size()){
      if(CLS_TABLE[i][0][4]>max)
        max = CLS_TABLE[i][0][4];
    }
  }
  return max;
}