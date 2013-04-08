// proof of idea of doing multiple scans at once with tuples
// $Id$

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include "../slinktest/src/NodeUtils.hh"

using namespace std;

template <typename V>
void print_dvec(const char *name, const V &v)
{
    printf("%s = {", name);
    for (int i=0; i<v.size(); i++) {
	int hi = v[i];
	printf("%d", hi);
	if (i<v.size()-1) printf(", ");
	else printf("}\n");
    }
}

struct sumTuple
{
    template <typename Tuple>
    __host__ __device__
    Tuple operator()(const Tuple &lhs, const Tuple &rhs) {
	using thrust::get;
	return thrust::make_tuple(get<0>(lhs) + get<0>(rhs),
				  get<1>(lhs) + get<1>(rhs));
    }
};


int main(int argc, char **argv)
{
    int a[10] = {0, 0, 0, 0, 1, 0, 0, 1, 0, 1};
    int b[10] = {0, 1, 0, 0, 1, 0, 1, 1, 1, 0};

    thrust::device_vector<int> d_a(a, a+10);
    thrust::device_vector<int> d_b(b, b+10);
    thrust::device_vector<int> d_oa(10);
    thrust::device_vector<int> d_ob(10);
    
    thrust::inclusive_scan(d_a.begin(), d_a.end(), d_oa.begin());
    thrust::inclusive_scan(d_b.begin(), d_b.end(), d_ob.begin());
    printf("expected output:\n");
    print_dvec("d_oa", d_oa);
    print_dvec("d_ob", d_ob);

    thrust::inclusive_scan(
	thrust::make_zip_iterator(thrust::make_tuple(d_a.begin(), d_b.begin())),
	thrust::make_zip_iterator(thrust::make_tuple(d_a.end(), d_b.end())),
	thrust::make_zip_iterator(thrust::make_tuple(d_oa.begin(), d_ob.begin())),
	sumTuple());
    
    printf("output:\n");
    print_dvec("d_oa", d_oa);
    print_dvec("d_ob", d_ob);
    
}
