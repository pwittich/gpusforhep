//
// system headers
//
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sched.h>
#include <vector>
#include <linux/types.h>

#include "NodeUtils.hh"

using namespace std;

//
// s32pci64 headers
//
extern "C" {
#include <s32pci64-filar.h>
#include <s32pci64-solar.h>
}

#define NLAYER 6
#define SI_REGIONS 3

//structure for hit combinations
struct Comb{
  int Hit[NLAYER];
};


//structure for track information
struct Track{
  int PT   ;
  int PHI  ;
  int D0   ;
  int CHI  ;
};

//structure for the constant set
struct ConstSet{
  int CHI0[NLAYER];
  int CHI1[NLAYER];
  int CHI2[NLAYER];

  int PT[NLAYER] ;
  int D0[NLAYER] ;
  int PHI[NLAYER];

  int IC_PT ;
  int IC_D0 ;
  int IC_PHI;

  int IC_CHI0;
  int IC_CHI1;
  int IC_CHI2;

};

//Initialize constant sets, that will be accessed by the track fitting
ConstSet Global_ConstSet[SI_REGIONS];

void InitGlobalConst();
void InitLocalConst(struct ConstSet*, int);
void FitFunc(struct Comb*, struct ConstSet*, struct Track*, int);
Track TrkFit(unsigned*,int);
void PrintConstSet(struct ConstSet*);

