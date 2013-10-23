

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#include <CL/cl.hpp>

#include "helperFuncs.h"
#include "clHelperFuncs.h"

using namespace std;

//get all platforms and devices and return index for "best" GPU
void CL_HELPERFUNCS::getPlatformsAndDevices(cl::vector< cl::Platform > *platformList,
		cl::vector< cl::vector< cl::Device > *> *deviceList,
		unsigned int &GPU_platform_index, unsigned int &GPU_device_index)
{
	cl_int err;
	cl::Platform::get(platformList);

	checkErr((*platformList).size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");

	cout << __FANCY__ << "Num of Platforms: " << (*platformList).size() << endl;

	string platformVendor;
	string platformVersion;
	cl_device_type deviceType;
	GPU_device_index = -1; GPU_platform_index = -1;

	//loop through platforms and devices
	for(unsigned int i=0;i<(*platformList).size();++i)
	{
		(*platformList)[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
		cout << __FANCY__ << "Platform " << i << " is by: " << platformVendor << "\n";

		(*platformList)[i].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &platformVersion);
		cout << __FANCY__ << "Platform " << i << " version is: " << platformVersion << "\n";

		(*deviceList).push_back(new cl::vector< cl::Device >());

		//int num_devices;
		(*platformList)[i].getDevices(CL_DEVICE_TYPE_ALL,(*deviceList)[i]);

		cout << __FANCY__ << "Platform " << i << " number of Devices: " << (*deviceList)[i]->size() << endl;

		for(unsigned int j=0;j<(*deviceList)[i]->size();++j)
		{
			deviceType = (*((*deviceList)[i]))[j].getInfo<CL_DEVICE_TYPE>(&err);
			checkErr(err, "get DEVICE TYPE error");
			cout << __FANCY__ << "\tDevice " << j << " is " <<
					(	(deviceType==CL_DEVICE_TYPE_GPU)?"GPU":
						((deviceType==CL_DEVICE_TYPE_CPU)?"CPU":"Unknown")) << " type " << endl;

			if (deviceType==CL_DEVICE_TYPE_GPU) {
				GPU_device_index = j; GPU_platform_index = i;
			}
			//print out info
			cout << __FANCY__ << "\t   Device name is " <<
				(*((*deviceList)[i]))[j].getInfo<CL_DEVICE_VENDOR>(&err) << ", " <<
				(*((*deviceList)[i]))[j].getInfo<CL_DEVICE_NAME>(&err) <<           endl;
			cout << __FANCY__ << "\t   Max work group size " <<
				(*((*deviceList)[i]))[j].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err) << endl;
			cout << __FANCY__ << "\t   Compute cores " <<
				(*((*deviceList)[i]))[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err) << endl;
			cout << __FANCY__ << "\t   Local memory (KB) " <<
				(((*((*deviceList)[i]))[j].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err))>>10) << endl;
			cout << __FANCY__ << "\t   Global memory (MB) " <<
				(((*((*deviceList)[i]))[j].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(&err))>>20) << endl;
			cout << __FANCY__ << "\t   Global memory cache (KB) " <<
				(((*((*deviceList)[i]))[j].getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>(&err))>>10) << endl;

			checkErr(err, "get info error");

		}
	}

}

bool CL_HELPERFUNCS::isDeviceTypeGPU(cl::vector< cl::vector< cl::Device > *> *deviceList,
		unsigned int GPU_platform_index, unsigned int GPU_device_index)
{
	if(GPU_platform_index >= (*deviceList).size() || //check valid indices
			GPU_device_index >= (*deviceList)[GPU_platform_index]->size()) return false;

	cl_int err;
	return CL_DEVICE_TYPE_GPU == (*((*deviceList)[GPU_platform_index]))[GPU_device_index].getInfo<CL_DEVICE_TYPE>(&err);
}

void CL_HELPERFUNCS::displayBuildLog(cl::Program &program, cl::Device &device)
{
        //Shows the build log

        char* build_log;
	size_t log_size;
        
	// First call to know the proper size
	clGetProgramBuildInfo(program(),device(), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	if(log_size < 3) //success
	  {  cout << __FANCY__ << "No build log.. Likely Successful!" << endl << endl; return; }
	
	build_log = new char[log_size+1]; //allocate log memory

	// Second call to get the log
	clGetProgramBuildInfo(program(),device(), CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);

	build_log[log_size] = '\0';
	cout << __FANCY__ << build_log << endl;

	delete[] build_log;  //release log memory
}
