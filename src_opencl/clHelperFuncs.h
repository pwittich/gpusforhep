/*
 * clHelperFuncs.h
 *
 *  Created on: Feb 28, 2013
 *      Author: Ryan Rivera
 */

#ifndef CL_HELPERFUNCS_H_
#define CL_HELPERFUNCS_H_

namespace CL_HELPERFUNCS
{
  //checkErr ~~
  //	used to check and output OpenCL errors
  inline void checkErr(cl_int e, const char * name)
  {
    if (e != CL_SUCCESS)
      {
	std::cerr << __FANCY__ << "ERROR: " << name
		  << " (" << e << ")" << std::endl;
	exit(EXIT_FAILURE);
      }
  }

  //getPlatformsAndDevices ~~
  //	get vector of all platforms and devices for each platform
  //	also return by reference the indices for the "best" GPU
  void getPlatformsAndDevices(cl::vector< cl::Platform > *platformList,
			      cl::vector< cl::vector< cl::Device > *> *deviceList,
			      unsigned int &GPU_platform_index, unsigned int &GPU_device_index);

  //isDeviceTypeGPU ~~
  //	return true if specified device is of type GPU ( CL_DEVICE_TYPE_GPU )
  bool isDeviceTypeGPU(cl::vector< cl::vector< cl::Device > *> *deviceList,
		       unsigned int GPU_platform_index, unsigned int GPU_device_index);

  void displayBuildLog(cl::Program &program, cl::Device &device);
  
}

#endif /* CLHELPERFUNCS_H_ */
