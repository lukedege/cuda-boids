#pragma once

#include <vector>

#include <cuda_gl_interop.h>

#include "cuda_utils.cuh"

namespace utils::cuda
{
	class gl_manager
	{
	public:
		// Returns a device pointer to the newly acquired resource for CUDA kernels' usage
		void* add_resource(unsigned int buffer_object, cudaGraphicsMapFlags flags)
		{
			// register this buffer object with CUDA
			cudaGraphicsResource* new_resource;
			checks::check_cuda_result(cudaGraphicsGLRegisterBuffer(&new_resource, buffer_object, flags));

			// map OpenGL buffer object for writing from CUDA
			void* device_ptr;
			checks::check_cuda_result(cudaGraphicsMapResources(1, &new_resource, 0));
			size_t num_bytes;

			checks::check_cuda_result(cudaGraphicsResourceGetMappedPointer(&device_ptr, &num_bytes, new_resource));
			std::cout << "Successfully CUDA mapped buffer: May access" << num_bytes << " bytes\n";

			resources.push_back(new_resource);

			return device_ptr;
		}

		~gl_manager()
		{
			checks::check_cuda_result(cudaGraphicsUnmapResources(resources.size(), resources.data()));
			for (auto res_ptr : resources)
			{
				checks::check_cuda_result(cudaGraphicsUnregisterResource(res_ptr));
			}
		}

	private:
		std::vector<cudaGraphicsResource*> resources;
	};
}