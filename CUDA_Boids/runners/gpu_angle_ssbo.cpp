#include "gpu_angle_ssbo.h"

// std libraries
#include <iostream>
#include <vector>
#include <math.h>

#include <glad.h>
#include <glm/glm.hpp>
#include <glm/gtx/vector_angle.hpp> 

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

// utils libraries
#include "../utils/CUDA/vector_math.h"
#include "../utils/CUDA/cuda_utils.cuh"
#include "../utils/utils.h"

namespace
{
	__global__ void kernel(float2* ssbo_positions, float* ssbo_angles, size_t size, const float delta_time)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		float2 vel{ 1,1 };
		float2 vel_norm{ normalize(vel) };
		float2 ref{ 1,0 };
		//printf("%f, %f, %f\n", delta_time, ssbo_positions[i].x, ssbo_positions[i].y);
		if (i < size)
		{
			ssbo_angles[i] = acos(clamp(dot(vel_norm, ref), -1.f, 1.f));
			ssbo_positions[i] += vel * delta_time;
		}
	}
}

namespace utils::runners
{
	gpu_angle_ssbo::gpu_angle_ssbo(const size_t amount) :
		shader { "shaders/ssbo_instanced_angle.vert", "shaders/basic.frag"},
		amount{ amount },
		triangles{ setup_mesh(), amount },
		block_size{ 32 },
		grid_size{ static_cast<size_t>(utils::cuda::math::ceil(amount, block_size)) },
		ssbo_positions_dptr{ nullptr },
		ssbo_angles_dptr{ nullptr }
	{
		GLuint ssbo_positions; // shader_storage_buffer_object
		GLuint ssbo_angles; // shader_storage_buffer_object

		std::vector<float> angles(amount);

		utils::containers::random_vec2_fill_cpu(triangles.positions, -20, 20);

		setup_ssbo(ssbo_positions, sizeof(glm::vec2), amount, 0, triangles.positions.data());
		setup_ssbo(ssbo_angles   , sizeof(float)    , amount, 1, 0);

		ssbo_positions_dptr = (float2*)cuda_gl_manager.add_resource(ssbo_positions, cudaGraphicsMapFlagsNone);
		ssbo_angles_dptr = (float*)cuda_gl_manager.add_resource(ssbo_angles, cudaGraphicsMapFlagsNone);
	}

	void gpu_angle_ssbo::calculate(const float delta_time)
	{
		kernel CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_angles_dptr, amount, delta_time);
		cudaDeviceSynchronize();
	}

	void gpu_angle_ssbo::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		shader.use();
		shader.setMat4("view_matrix", view_matrix);
		shader.setMat4("projection_matrix", projection_matrix);
		triangles.draw(shader, view_matrix);
	}
}