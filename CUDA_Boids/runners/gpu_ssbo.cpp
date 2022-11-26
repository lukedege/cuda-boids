#include "gpu_ssbo.h"

// std libraries
#include <vector>
#include <math.h>

#include <glm/glm.hpp>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

// utils libraries
#include "../utils/CUDA/vector_math.h"
#include "../utils/CUDA/cuda_utils.cuh"
#include "../utils/utils.h"
#include "boid_behaviours.h"

namespace
{
	__global__ void alignment      (float4* alignments     , float4* positions, float4* velocities, size_t amount, size_t max_radius) {/*TODO*/}
	__global__ void cohesion       (float4* cohesions      , float4* positions, size_t amount, size_t max_radius)                     {/*TODO*/}
	__global__ void separation     (float4* separations    , float4* positions, size_t amount, size_t max_radius)                     {/*TODO*/}
	__global__ void wall_separation(float4* wall_separation, float4* positions, utils::math::plane* borders, size_t amount)           {/*TODO*/}

	__global__ void blender(float4* ssbo_positions, float4* ssbo_velocities, 
		float4* alignments, float4* cohesions, float4* separations, float4* wall_separations,
		utils::runners::boid_runner::simulation_parameters simulation_params, size_t amount, const float delta_time)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		float4 accel_blend;
		
		accel_blend =  simulation_params.alignment_coeff       * alignments[i]
			         + simulation_params.cohesion_coeff        * cohesions[i]
			         + simulation_params.separation_coeff      * separations[i]
			         + simulation_params.wall_separation_coeff * wall_separations[i];
		
		ssbo_velocities[i] = normalize(ssbo_velocities[i] + accel_blend * delta_time); //v = u + at
		ssbo_positions [i] += ssbo_velocities[i] * simulation_params.boid_speed * delta_time; //s = vt
	}

}

namespace utils::runners
{
	gpu_vel_ssbo::gpu_vel_ssbo() :
		shader{ "shaders/ssbo.vert", "shaders/basic.frag"},
		amount{ sim_params.boid_amount },
		triangle_mesh{setup_mesh()},
		positions { std::vector<glm::vec4>(amount) },// TODO unified memory or transfers
		velocities{ std::vector<glm::vec4>(amount) },// TODO unified memory or transfers
		block_size{ 32 },
		grid_size{ static_cast<size_t>(utils::cuda::math::ceil(amount, block_size)) },
		ssbo_positions_dptr { nullptr },
		ssbo_velocities_dptr{ nullptr }
	{
		// shader storage buffer objects
		GLuint ssbo_positions;
		GLuint ssbo_velocities;

		utils::containers::random_vec4_fill_cpu(positions, -20, 20);
		utils::containers::random_vec4_fill_cpu(velocities, -1, 1);

		setup_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 0, positions.data());
		setup_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 1, velocities.data());

		ssbo_positions_dptr  = (float4*)cuda_gl_manager.add_resource(ssbo_positions , cudaGraphicsMapFlagsNone);
		ssbo_velocities_dptr = (float4*)cuda_gl_manager.add_resource(ssbo_velocities, cudaGraphicsMapFlagsNone);

		//cudaMalloc is sufficient since no transfers between CPU-GPU (no cudaMemcpy)
		cudaMalloc(&alignments_dptr      , amount * sizeof(float4));
		cudaMalloc(&cohesions_dptr       , amount * sizeof(float4));
		cudaMalloc(&separations_dptr     , amount * sizeof(float4));
		cudaMalloc(&wall_separations_dptr, amount * sizeof(float4));
	}

	void gpu_vel_ssbo::calculate(const float delta_time)
	{
		alignment       CUDA_KERNEL(grid_size, block_size)(alignments_dptr      ,ssbo_positions_dptr, ssbo_velocities_dptr, amount, sim_params.boid_fov/*TODO to put in device first*/);
		cohesion        CUDA_KERNEL(grid_size, block_size)(cohesions_dptr       ,ssbo_positions_dptr, amount, sim_params.boid_fov/*TODO to put in device first*/);
		separation      CUDA_KERNEL(grid_size, block_size)(separations_dptr     ,ssbo_positions_dptr, amount, sim_params.boid_fov/*TODO to put in device first*/);
		wall_separation CUDA_KERNEL(grid_size, block_size)(wall_separations_dptr,ssbo_positions_dptr, simulation_volume_planes.data()/*TODO to put in device first*/, amount);
		cudaDeviceSynchronize();
		//blend em
		blender CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, alignments_dptr, cohesions_dptr, separations_dptr, wall_separations_dptr, sim_params/*TODO to put in device first*/, amount, delta_time);
		cudaDeviceSynchronize();
	}

	void gpu_vel_ssbo::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		shader.use();
		shader.setMat4("view_matrix", view_matrix);
		shader.setMat4("projection_matrix", projection_matrix);

		triangle_mesh.draw_instanced(amount);
	}

	gpu_vel_ssbo::~gpu_vel_ssbo()
	{
		cudaFree(&alignments_dptr      );
		cudaFree(&cohesions_dptr       );
		cudaFree(&separations_dptr     );
		cudaFree(&wall_separations_dptr);
	}

	gpu_vel_ssbo::simulation_parameters gpu_vel_ssbo::get_simulation_parameters()
	{
		//TODO
		return {};
	}

	void gpu_vel_ssbo::set_simulation_parameters(simulation_parameters new_params)
	{
		//TODO
	}
}