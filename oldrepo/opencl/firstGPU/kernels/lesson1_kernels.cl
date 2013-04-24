#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant char hw[] = "Hello Werld\n";
__kernel void hello(__global DATAWORD *in, __global DATAWORD * out)
{
	size_t tid = get_global_id(0);
	if(tid%2==0)
	  out[tid] = in[tid];
	else if(tid%2==1)
	  out[tid] = hw[tid];
}