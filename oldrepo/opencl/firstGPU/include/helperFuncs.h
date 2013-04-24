/*
 * helperFuncs.h
 *
 *  Created on: Feb 28, 2013
 *      Author: Ryan Rivera
 */

#ifndef HELPERFUNCS_H_
#define HELPERFUNCS_H_

#include <iostream> //for std:cerr
#include <cstdlib> //for EXIT_FAILURE

namespace HELPERFUNCS
{
  #define S(x) #x			//convert numbers to string
  #define S_(x) S(x)
  #define __FANCY__ (std::string)((std::string)(__FUNCTION__) + (std::string)("():[") + (std::string)(S_(__LINE__)) + "] ")

  //initMainAppStatus ~~
  //	setup main process
  //		enableMask: bit-wise enable
  //			0-sysPriority, 1-cpuAffinity, 2-sched, 3-threadSched
  //		sysPriority: max = -20 (same as nice()?)
  //		cpuWithAffinity: index of CPU to run on and block interrupts
  //		schedPolicy: SCHED_FIFO,_RR,_OTHER
  //		schedPriority: max = 99
  //		threadSchedPolicy: SCHED_FIFO,_RR,_OTHER
  //		threadSchedPriority: max = 99
  #define APP_STATUS_SYS_PRIORITY 	1
  #define APP_STATUS_CPU_AFFINITY 		1<<1
  #define APP_STATUS_SCHED 			1<<2
  #define APP_STATUS_THREAD_SCHED 		1<<3
  void initMainAppStatus(
			 int enableMask,
			 int sysPriority=0,
			 int cpuWithAffinity=0,
			 int schedPolicy=0,
			 int schedPriority=0,
			 int threadSchedPolicy=0,
			 int threadSchedPriority=0
			 );


  // The following defines specialized templates to provide a string
  // containing the typename
  template<class T>
    struct TemplateDataType
    {
      std::string  getType();
      unsigned int getSize() {return sizeof(T);}
    private:
      T *t;
    };

}



#endif /* HELPERFUNCS_H_ */
