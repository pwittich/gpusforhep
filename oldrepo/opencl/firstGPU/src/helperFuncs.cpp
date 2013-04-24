#include <CL/cl.hpp>
#include <cstdio>


//#include <utility>
#include <sys/resource.h>

#include "helperFuncs.h"

using namespace std;

void HELPERFUNCS::initMainAppStatus(
		int enableMask,
		int sysPriority,
		int cpuWithAffinity,
		int schedPolicy,
		int schedPriority,
		int threadSchedPolicy,
		int threadSchedPriority	)
{
	//------------------------------------------------------
	if(enableMask & APP_STATUS_SYS_PRIORITY)
	{
		//set top system priority (seems to have little effect on run-time)
		if(-1 ==  setpriority(PRIO_PROCESS,0,sysPriority))
			cerr << __FANCY__ << "failed system priority" << endl; //highest priority is -20 (-20 to 19)
	}
	cerr << __FANCY__ << "System Priority: " << getpriority(PRIO_PROCESS,0) << endl;

	//------------------------------------------------------
	cpu_set_t mask;
	if(enableMask & APP_STATUS_CPU_AFFINITY)
	{
		//set cpu affinity (had big effect on run-time)
		if (cpuWithAffinity > -1)
		{
			int status;

			CPU_ZERO(&mask);
			CPU_SET(cpuWithAffinity, &mask); //set bit in mask for certain processor
			status = sched_setaffinity(0, sizeof(mask), &mask);
			if (status != 0)
				cerr << __FANCY__ << "failed sched set affinity" << endl;
		}
		else
			cerr << __FANCY__ << "failed sched set affinity" << endl;

	}
	if (sched_getaffinity(0, sizeof(mask), &mask) >= 0)
		cerr << __FANCY__ << "Affinity mask is: 0x" <<  hex << *((unsigned long*)&mask) << dec << endl;


	//------------------------------------------------------
	struct sched_param p;
	if(enableMask & APP_STATUS_SCHED)
	{
		//set scheduling policy (seems to increase timing spread, but can get lower average run times sporadically)
		//this also propogates to thread schedule policy
		p.sched_priority = schedPriority;
		if (sched_setscheduler(0, schedPolicy, &p))
			cerr << __FANCY__ << "failed sched set scheduler" << endl;
	}
	if (sched_getparam(0, &p) == 0)
		cerr << __FANCY__ << "Process scheduling priority=" << p.sched_priority << endl;


	//------------------------------------------------------
	int policy;
	if(enableMask & APP_STATUS_THREAD_SCHED)
	{
		//set thread scheduling policy (seems to increase timing spread, but can get lower average run times sporadically)
		policy = threadSchedPolicy;
		p.sched_priority = threadSchedPriority; //max priority

		if (pthread_setschedparam(pthread_self(), policy, &p) != 0)
			cerr << __FANCY__ << "failed pthread_setschedparam" << endl;

	}
	if (pthread_getschedparam(pthread_self(), &policy, &p) == 0)
	{

		cerr << __FANCY__ << "Thread policy=" << ((policy == SCHED_FIFO)  ? "SCHED_FIFO" :
							   (policy == SCHED_RR)    ? "SCHED_RR" :
							   (policy == SCHED_OTHER) ? "SCHED_OTHER" :
														 "???")
			  << ", priority=" << p.sched_priority << std::endl;
	}


}

namespace HELPERFUNCS
{
  template<> std::string TemplateDataType<unsigned long>::getType() {return(std::string("ulong")); }
  template<> std::string TemplateDataType<double       >::getType() {return(std::string("double")); }
  template<> std::string TemplateDataType<float        >::getType() {return(std::string("float")); }
  template<> std::string TemplateDataType<long         >::getType() {return(std::string("long")); }
  template<> std::string TemplateDataType<unsigned int >::getType() {return(std::string("uint")); }
  template<> std::string TemplateDataType<int          >::getType() {return(std::string("int")); }
  template<> std::string TemplateDataType<unsigned char>::getType() {return(std::string("uchar")); }
  template<> std::string TemplateDataType<char         >::getType() {return(std::string("char")); }
}
