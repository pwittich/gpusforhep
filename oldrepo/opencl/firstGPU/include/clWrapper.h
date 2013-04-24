/*
 * clWrapper.h
 *
 *  Created on: Feb 28, 2013
 *      Author: Ryan Rivera
 */

#ifndef CL_WRAPPER_H_
#define CL_WRAPPER_H_




// The following defines specialized templates to provide a string
// containing the typename
template<class T>
struct TypeName {
  string getName();
private:
  T *t;
};

template<> string TypeName<double>::getName() {return(string("double")); }
template<> string TypeName<float>::getName() {return(string("float")); }
template<> string TypeName<unsigned long>::getName() {return(string("ulong"));}
template<> string TypeName<long>::getName() { return(string("long")); }
template<> string TypeName<unsigned int>::getName() {return(string("uint"));}
template<> string TypeName<int>::getName() {return(string("int")); }
template<> string TypeName<unsigned char>::getName() {return(string("uchar"));}
template<> string TypeName<char>::getName() {return(string("char")); }

// specification of the OclTest template
template <typename TYPE1>
class clWrapper {
private:

	string dataType;
	static unsigned int GPU_device_index, GPU_platform_index;

	static cl::vector< cl::Platform > platformList;
	static cl::vector< cl::vector< cl::Device > *> deviceList;


public:

	clWrapper()
	{
		dataType = TypeName<TYPE1>().getName();
		cout << __FANCY__ << "My data type is <" << dataType.c_str() << ">" << endl;

			unsigned int GPU_device_index, GPU_platform_index;

		if(platformList.size())
		{

			cout << __FANCY__ << "Already have platforms: " << platformList.size() << endl;
			cout << __FANCY__ << "GPU [platform,device] indices: [" << GPU_platform_index
					<< "," << GPU_device_index << "]" << endl;
			return;
		}

		getPlatformsAndDevices(&platformList,&deviceList,GPU_device_index,GPU_platform_index);

		cout << __FANCY__ << "GPU [platform,device] indices: [" << GPU_platform_index
				<< "," << GPU_device_index << "]" << endl;
	}

	void getKernel(int i); //get index within directory kernel

	void getKernel(char * fn); //get index within directory kernel



};


template <typename TYPE1>
cl::vector< cl::Platform > clWrapper<TYPE1>::platformList;

template <typename TYPE1>
cl::vector< cl::vector< cl::Device > *> clWrapper<TYPE1>::deviceList;

template <typename TYPE1>
unsigned int clWrapper<TYPE1>::GPU_device_index = -1;

template <typename TYPE1>
unsigned int clWrapper<TYPE1>::GPU_platform_index = -1;

#endif /* CL_WRAPPER_H_ */
