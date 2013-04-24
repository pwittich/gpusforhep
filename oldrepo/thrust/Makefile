CUDA_DIR=/opt/cuda
CUDA_SDK_DIR=/opt/cuda/sdk

INCFLAG=-I $(CUDA_DIR)/include -I$(CUDA_SDK_DIR)/C/common/inc -I..


# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM20) #$(GENCODE_SM30)

NVCC = nvcc
#NVCCFLAGS = -O3 $(INCFLAG) --ptxas-options=-v -use_fast_math $(GENCODE_FLAGS)
#NVCCFLAGS = -g $(INCFLAG)  --use_fast_math $(GENCODE_SM20) -DTHRUST_DEBUG
NVCCFLAGS = -O3 $(INCFLAG)  --use_fast_math $(GENCODE_SM20) 

LIBS=-L/home/wittich/src/cudpp_install_2.0/lib -lcudpp

%.o: %.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: markpack

markpack: markpack.o 
	$(NVCC) $^ -o $@ 

mp2: mp2.o 
	$(NVCC) $^ -o $@ 

unpack: unpack.o 
	$(NVCC) $^ -o $@ 

get_roads: get_roads.o 
	$(NVCC) $^ -o $@ 

clean:
	$(RM) *.o *~

depend:
	makedepend -Y $(INCFLAG) *.cu *.cc *.h

# DO NOT DELETE

markpack.o: /opt/cuda/include/thrust/host_vector.h
markpack.o: /opt/cuda/include/thrust/detail/config.h
markpack.o: /opt/cuda/include/thrust/detail/config/config.h
markpack.o: /opt/cuda/include/thrust/detail/config/simple_defines.h
markpack.o: /opt/cuda/include/thrust/detail/config/compiler.h
markpack.o: /opt/cuda/include/thrust/detail/config/device_backend.h
markpack.o: /opt/cuda/include/thrust/detail/config/host_device.h
markpack.o: /opt/cuda/include/host_defines.h
markpack.o: /opt/cuda/include/thrust/detail/config/debug.h
markpack.o: /opt/cuda/include/thrust/detail/config/compiler_fence.h
markpack.o: /opt/cuda/include/thrust/detail/vector_base.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/normal_iterator.h
markpack.o: /opt/cuda/include/thrust/iterator/iterator_adaptor.h
markpack.o: /opt/cuda/include/thrust/iterator/iterator_facade.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits/has_trivial_assign.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/iterator_facade.inl
markpack.o: /opt/cuda/include/thrust/iterator/iterator_categories.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/universal_categories.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/backend_iterator_categories.h
markpack.o: /opt/cuda/include/thrust/iterator/iterator_traits.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/any_space_tag.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/backend_iterator_spaces.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/iterator_traits.inl
markpack.o: /opt/cuda/include/thrust/iterator/detail/iterator_category_to_traversal.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/iterator_category_to_space.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/device_iterator_category_to_backend_space.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/is_iterator_category.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/distance_from_result.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/iterator_adaptor.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/dereference.h
markpack.o: /opt/cuda/include/thrust/device_ptr.h
markpack.o: /opt/cuda/include/thrust/detail/pointer_base.h
markpack.o: /opt/cuda/include/thrust/detail/pointer_base.inl
markpack.o: /opt/cuda/include/thrust/detail/type_traits/pointer_traits.h
markpack.o: /opt/cuda/include/thrust/detail/device_ptr.inl
markpack.o: /opt/cuda/include/thrust/device_reference.h
markpack.o: /opt/cuda/include/thrust/detail/reference_base.h
markpack.o: /opt/cuda/include/thrust/detail/reference_base.inl
markpack.o: /opt/cuda/include/thrust/detail/copy.h
markpack.o: /opt/cuda/include/thrust/detail/copy.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/copy.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/dispatch/copy.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/minimum_space.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits/minimum_type.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/dispatch/copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/detail/general_copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/detail/trivial_copy.h
markpack.o: /opt/cuda/include/thrust/detail/dispatch/is_trivial_copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/dispatch/copy.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/minimum_category.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/copy_device_to_device.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/copy_device_to_device.inl
markpack.o: /opt/cuda/include/thrust/distance.h
markpack.o: /opt/cuda/include/thrust/detail/distance.inl
markpack.o: /opt/cuda/include/thrust/detail/dispatch/distance.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/copy_host_or_any_to_device.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/copy_host_or_any_to_device.inl
markpack.o: /opt/cuda/include/thrust/detail/static_assert.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/copy_device_to_host_or_any.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/copy_device_to_host_or_any.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/dispatch/copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/copy_cross_space.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/copy_cross_space.inl
markpack.o: /opt/cuda/include/thrust/detail/uninitialized_array.h
markpack.o: /opt/cuda/include/thrust/detail/backend/internal_allocator.h
markpack.o: /opt/cuda/include/thrust/device_malloc.h
markpack.o: /opt/cuda/include/thrust/detail/device_malloc.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/dispatch/malloc.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/malloc.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/malloc.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/malloc.inl
markpack.o: /opt/cuda/include/cuda_runtime_api.h
markpack.o: /opt/cuda/include/builtin_types.h
markpack.o: /opt/cuda/include/device_types.h /opt/cuda/include/driver_types.h
markpack.o: /opt/cuda/include/surface_types.h
markpack.o: /opt/cuda/include/texture_types.h
markpack.o: /opt/cuda/include/vector_types.h
markpack.o: /opt/cuda/include/thrust/system/cuda_error.h
markpack.o: /opt/cuda/include/thrust/system/error_code.h
markpack.o: /opt/cuda/include/thrust/system/detail/errno.h
markpack.o: /opt/cuda/include/thrust/system/detail/error_category.inl
markpack.o: /opt/cuda/include/thrust/functional.h
markpack.o: /opt/cuda/include/thrust/detail/functional/placeholder.h
markpack.o: /opt/cuda/include/thrust/detail/functional/actor.h
markpack.o: /opt/cuda/include/thrust/tuple.h
markpack.o: /opt/cuda/include/thrust/detail/tuple.inl
markpack.o: /opt/cuda/include/thrust/pair.h
markpack.o: /opt/cuda/include/thrust/detail/pair.inl
markpack.o: /opt/cuda/include/thrust/detail/functional/value.h
markpack.o: /opt/cuda/include/thrust/detail/functional/composite.h
markpack.o: /opt/cuda/include/thrust/detail/functional/operators/assignment_operator.h
markpack.o: /opt/cuda/include/thrust/detail/functional/operators/operator_adaptors.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits/result_of.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits/function_traits.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits/has_nested_type.h
markpack.o: /opt/cuda/include/thrust/detail/functional/actor.inl
markpack.o: /opt/cuda/include/thrust/detail/functional/argument.h
markpack.o: /opt/cuda/include/thrust/detail/functional.inl
markpack.o: /opt/cuda/include/thrust/detail/functional/operators.h
markpack.o: /opt/cuda/include/thrust/detail/functional/operators/arithmetic_operators.h
markpack.o: /opt/cuda/include/thrust/detail/functional/operators/relational_operators.h
markpack.o: /opt/cuda/include/thrust/detail/functional/operators/logical_operators.h
markpack.o: /opt/cuda/include/thrust/detail/functional/operators/bitwise_operators.h
markpack.o: /opt/cuda/include/thrust/detail/functional/operators/compound_assignment_operators.h
markpack.o: /opt/cuda/include/thrust/detail/functional/operators/comma_operator.h
markpack.o: /opt/cuda/include/thrust/system/detail/error_code.inl
markpack.o: /opt/cuda/include/thrust/system/detail/error_condition.inl
markpack.o: /opt/cuda/include/thrust/system/detail/cuda_error.inl
markpack.o: /opt/cuda/include/thrust/system/detail/bad_alloc.h
markpack.o: /opt/cuda/include/thrust/detail/backend/no_throw_free.h
markpack.o: /opt/cuda/include/thrust/detail/backend/dispatch/no_throw_free.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/no_throw_free.h
markpack.o: /opt/cuda/include/thrust/device_free.h
markpack.o: /opt/cuda/include/thrust/detail/device_free.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/dispatch/free.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/free.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/free.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/free.inl
markpack.o: /opt/cuda/include/thrust/system_error.h
markpack.o: /opt/cuda/include/thrust/system/system_error.h
markpack.o: /opt/cuda/include/thrust/system/detail/system_error.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/no_throw_free.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/no_throw_free.inl
markpack.o: /opt/cuda/include/thrust/detail/contiguous_storage.h
markpack.o: /opt/cuda/include/thrust/detail/contiguous_storage.inl
markpack.o: /opt/cuda/include/thrust/swap.h
markpack.o: /opt/cuda/include/thrust/detail/swap.inl
markpack.o: /opt/cuda/include/thrust/detail/swap_ranges.inl
markpack.o: /opt/cuda/include/thrust/iterator/zip_iterator.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/zip_iterator_base.h
markpack.o: /opt/cuda/include/thrust/detail/tuple_meta_transform.h
markpack.o: /opt/cuda/include/thrust/detail/tuple_transform.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/zip_iterator.inl
markpack.o: /opt/cuda/include/thrust/detail/internal_functional.h
markpack.o: /opt/cuda/include/thrust/for_each.h
markpack.o: /opt/cuda/include/thrust/detail/for_each.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/for_each.h
markpack.o: /opt/cuda/include/thrust/detail/backend/for_each.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/for_each.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/for_each.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/for_each.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/for_each.inl
markpack.o: /opt/cuda/include/thrust/detail/minmax.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/detail/launch_closure.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/arch.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/arch.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/detail/launch_closure.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/detail/launch_calculator.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/detail/launch_calculator.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/for_each.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/for_each.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/for_each.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/for_each.inl
markpack.o: /opt/cuda/include/thrust/detail/uninitialized_array.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/trivial_copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/trivial_copy.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/copy_device_to_device.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/copy_device_to_device.inl
markpack.o: /opt/cuda/include/thrust/transform.h
markpack.o: /opt/cuda/include/thrust/detail/transform.inl
markpack.o: /opt/cuda/include/thrust/detail/device_reference.inl
markpack.o: /opt/cuda/include/thrust/iterator/reverse_iterator.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/reverse_iterator_base.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/reverse_iterator.inl
markpack.o: /opt/cuda/include/thrust/detail/vector_base.inl
markpack.o: /opt/cuda/include/thrust/detail/overlapped_copy.h
markpack.o: /opt/cuda/include/thrust/equal.h
markpack.o: /opt/cuda/include/thrust/detail/equal.inl
markpack.o: /opt/cuda/include/thrust/mismatch.h
markpack.o: /opt/cuda/include/thrust/detail/mismatch.inl
markpack.o: /opt/cuda/include/thrust/find.h
markpack.o: /opt/cuda/include/thrust/detail/find.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/find.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/find.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/find.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/reduce.h
markpack.o: /opt/cuda/include/thrust/detail/backend/reduce.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/reduce.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits/algorithm/intermediate_type_from_function_and_iterators.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits/iterator/is_output_iterator.h
markpack.o: /opt/cuda/include/thrust/detail/type_traits/is_metafunction_defined.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/reduce.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/reduce.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/extern_shared_ptr.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/block/reduce.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/reduce_by_key.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/reduce_by_key.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/reduce.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/reduce.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/reduce_intervals.h
markpack.o: /opt/cuda/include/thrust/detail/backend/reduce_intervals.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/reduce_intervals.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/reduce_intervals.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/reduce_intervals.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/reduce_intervals.h
markpack.o: /opt/cuda/include/thrust/detail/backend/default_decomposition.h
markpack.o: /opt/cuda/include/thrust/detail/backend/decompose.h
markpack.o: /opt/cuda/include/thrust/detail/backend/default_decomposition.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/default_decomposition.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/default_decomposition.h
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/default_decomposition.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/default_decomposition.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/default_decomposition.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/reduce_by_key.h
markpack.o: /opt/cuda/include/thrust/extrema.h
markpack.o: /opt/cuda/include/thrust/detail/extrema.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/extrema.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/extrema.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/extrema.inl
markpack.o: /opt/cuda/include/thrust/reduce.h
markpack.o: /opt/cuda/include/thrust/detail/reduce.inl
markpack.o: /opt/cuda/include/thrust/transform_reduce.h
markpack.o: /opt/cuda/include/thrust/detail/transform_reduce.inl
markpack.o: /opt/cuda/include/thrust/iterator/transform_iterator.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/transform_iterator.inl
markpack.o: /opt/cuda/include/thrust/iterator/counting_iterator.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/counting_iterator.inl
markpack.o: /opt/cuda/include/thrust/detail/numeric_traits.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/extrema.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/find.h
markpack.o: /opt/cuda/include/thrust/uninitialized_fill.h
markpack.o: /opt/cuda/include/thrust/detail/uninitialized_fill.inl
markpack.o: /opt/cuda/include/thrust/detail/dispatch/uninitialized_fill.h
markpack.o: /opt/cuda/include/thrust/fill.h
markpack.o: /opt/cuda/include/thrust/detail/fill.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/fill.h
markpack.o: /opt/cuda/include/thrust/detail/backend/fill.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/fill.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/fill.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/fill.inl
markpack.o: /opt/cuda/include/thrust/detail/util/align.h
markpack.o: /opt/cuda/include/thrust/detail/cstdint.h
markpack.o: /opt/cuda/include/thrust/generate.h
markpack.o: /opt/cuda/include/thrust/detail/generate.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/fill.h
markpack.o: /opt/cuda/include/thrust/uninitialized_copy.h
markpack.o: /opt/cuda/include/thrust/detail/uninitialized_copy.inl
markpack.o: /opt/cuda/include/thrust/detail/dispatch/uninitialized_copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/uninitialized_copy.h
markpack.o: /opt/cuda/include/thrust/detail/backend/dispatch/uninitialized_copy.h
markpack.o: /opt/cuda/include/thrust/advance.h
markpack.o: /opt/cuda/include/thrust/detail/advance.inl
markpack.o: /opt/cuda/include/thrust/detail/dispatch/advance.h
markpack.o: /opt/cuda/include/thrust/detail/destroy.h
markpack.o: /opt/cuda/include/thrust/detail/dispatch/destroy.h
markpack.o: /opt/cuda/include/thrust/detail/host_vector.inl
markpack.o: /opt/cuda/include/thrust/device_vector.h
markpack.o: /opt/cuda/include/thrust/device_malloc_allocator.h
markpack.o: /opt/cuda/include/thrust/detail/device_vector.inl
markpack.o: /opt/cuda/include/thrust/copy.h
markpack.o: /opt/cuda/include/thrust/detail/copy_if.h
markpack.o: /opt/cuda/include/thrust/detail/copy_if.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/copy_if.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/copy_if.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/copy_if.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/scan.h
markpack.o: /opt/cuda/include/thrust/detail/backend/scan.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/scan.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/scan.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/scan.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/dispatch/scan.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/detail/fast_scan.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/detail/fast_scan.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/detail/safe_scan.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/detail/safe_scan.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/omp/scan.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/scan_by_key.h
markpack.o: /opt/cuda/include/thrust/detail/backend/generic/scan_by_key.inl
markpack.o: /opt/cuda/include/thrust/replace.h
markpack.o: /opt/cuda/include/thrust/detail/replace.inl
markpack.o: /opt/cuda/include/thrust/scatter.h
markpack.o: /opt/cuda/include/thrust/detail/scatter.inl
markpack.o: /opt/cuda/include/thrust/iterator/permutation_iterator.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/permutation_iterator_base.h
markpack.o: /opt/cuda/include/thrust/iterator/detail/permutation_iterator.inl
markpack.o: /opt/cuda/include/thrust/detail/backend/cpp/copy_if.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/copy_if.h
markpack.o: /opt/cuda/include/thrust/detail/backend/cuda/copy_if.inl
markpack.o: /opt/cuda/include/thrust/scan.h
markpack.o: /opt/cuda/include/thrust/detail/scan.inl
markpack.o: ../slinktest/src/NodeUtils.hh
