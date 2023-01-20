/*#include "gpu_vao.h"

// std libraries
#include <vector>
#include <math.h>

#include <glad.h>
#include <glm/glm.hpp>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

// utils libraries
#include "../utils/CUDA/vector_math.h"
#include "../utils/CUDA/cuda_utils.h"
#include "../utils/utils.h"


namespace
{
	__global__ void kernel(float3* vbo_positions, float3* vbo_velocities, size_t size, const float delta_time)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		float3 vel{ 1,1,0 };
		//printf("%f, %f, %f\n", delta_time, vbo_positions[i].x, vbo_positions[i].y);
		if (i < size)
		{
			vbo_velocities[i] = vel;
			vbo_positions[i] += vbo_velocities[i] * delta_time;
		}
	}
}

namespace utils::runners
{
	gpu_vao::gpu_vao(simulation_parameters params) :
		vao_runner{ { "shaders/vao.vert", "shaders/basic.frag", "shaders/vao.geom"}, params },
		amount{ params.static_params.boid_amount },
		positions { std::vector<glm::vec3>(amount) },
		velocities{ std::vector<glm::vec3>(amount) },
		block_size{ 32 },
		grid_size { static_cast<size_t>(utils::math::ceil(amount, block_size)) },
		vbo_positions_dptr { nullptr },
		vbo_velocities_dptr{ nullptr }
	{
		GLuint vbo_positions; 
		GLuint vbo_velocities;

		// TODO convert to vec4
		//utils::containers::random_vec3_fill_cpu(positions , -20, 20);
		//utils::containers::random_vec3_fill_cpu(velocities, -2, 2);

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		setup_vbo(vbo_positions,  sizeof(glm::vec3), positions .size(), 0, positions.data());
		setup_vbo(vbo_velocities, sizeof(glm::vec3), velocities.size(), 1, velocities.data());

		glBindVertexArray(0);
	
		vbo_positions_dptr  = (float3*)cuda_gl_manager.add_resource(vbo_positions, cudaGraphicsMapFlagsNone);
		vbo_velocities_dptr = (float3*)cuda_gl_manager.add_resource(vbo_velocities, cudaGraphicsMapFlagsNone);
	}
	
	void gpu_vao::calculate(const float delta_time)
	{
		kernel CUDA_KERNEL(grid_size, block_size)(vbo_positions_dptr, vbo_velocities_dptr, amount, delta_time);
		cudaDeviceSynchronize();
	}

	void gpu_vao::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		boid_shader.use();
		boid_shader.setMat4("view_matrix", view_matrix);
		boid_shader.setMat4("projection_matrix", projection_matrix);

		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, positions.size());
		glBindVertexArray(0);
	}
	
}*/