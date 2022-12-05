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
#include "../utils/CUDA/cuda_utils.h"
#include "../utils/utils.h"
#include "boid_behaviours.h"

namespace
{	
	__global__ void alignment      (float4* alignments     , float4* positions, float4* velocities, size_t amount, utils::runners::boid_runner::simulation_parameters* sim_params)
	{
		int current = blockIdx.x * blockDim.x + threadIdx.x;
		if (current >= amount) return; // avoid threads ids overflowing the array
		
		size_t max_radius = sim_params->boid_fov;// maybe passing the whole simparams is not that cool, maybe passing by value is just fine (we just need the fov)			
		float4 alignment{ 0 };
		bool in_radius;
		for (size_t i = 0; i < amount; i++)
		{
			// condition as multiplier avoids warp divergence
			in_radius = utils::math::distance2(positions[current], positions[i]) < max_radius * max_radius;
			alignment += velocities[i] * in_radius;
		}

		alignments[current] = utils::math::normalize_or_zero(alignment);
		
	}
	__global__ void cohesion       (float4* cohesions      , float4* positions, size_t amount, utils::runners::boid_runner::simulation_parameters* sim_params)
	{
		int current = blockIdx.x * blockDim.x + threadIdx.x;
		if (current >= amount) return;
		
		size_t max_radius = sim_params->boid_fov;
		float4 cohesion{ 0 }, baricenter{ 0 };
		float counter{ 0 };
		bool in_radius;
		for (size_t i = 0; i < amount; i++)
		{
			in_radius = utils::math::distance2(positions[current], positions[i]) < max_radius * max_radius;
			
			baricenter += positions[i] * in_radius;
			counter += 1.f * in_radius;
		}
		baricenter /= counter;
		cohesion = baricenter - positions[current];
		cohesions[current] = utils::math::normalize_or_zero(cohesion);
		
	}
	__global__ void separation     (float4* separations    , float4* positions, size_t amount, utils::runners::boid_runner::simulation_parameters* sim_params)
	{
		int current = blockIdx.x * blockDim.x + threadIdx.x;
		if (current >= amount) return;

		size_t max_radius = sim_params->boid_fov;
		float4 separation{ 0 };
		float4 repulsion;
		bool in_radius;

		// boid check
		for (size_t i = 0; i < amount; i++)
		{
			repulsion = positions[current] - positions[i];
			in_radius = utils::math::length2(repulsion) < max_radius * max_radius;
			separation += (utils::math::normalize_or_zero(repulsion) / (length(repulsion) + 0.0001f)) * in_radius; //TODO may be more optimizable but we'll see
		}

		separations[current] = utils::math::normalize_or_zero(separation);
	}
	__global__ void wall_separation(float4* wall_separations, float4* positions, utils::math::plane* borders, size_t amount)
	{
		int current = blockIdx.x * blockDim.x + threadIdx.x;
		if (current >= amount) return;
		
		float4 separation{ 0 };
		float4 repulsion;
		float4 plane_normal;
		float distance;
		float near_wall;
		// wall check
		for (size_t b = 0; b < 6; b++)
		{
			distance = utils::math::distance_point_plane(positions[current], borders[b]) + 0.0001f;
			near_wall = distance < 1.f;
			repulsion = (borders[b].normal / abs(distance)) * near_wall;
			separation += repulsion;
		}

		wall_separations[current] = utils::math::normalize_or_zero(separation);
	}

	__global__ void blender(float4* ssbo_positions, float4* ssbo_velocities, 
		float4* alignments, float4* cohesions, float4* separations, float4* wall_separations,
		utils::runners::boid_runner::simulation_parameters* simulation_params, size_t amount, const float delta_time)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= amount) return;
		
		float chs = simulation_params->cube_size / 2;
		float4 accel_blend;

		accel_blend = simulation_params->alignment_coeff * alignments[i]
			+ simulation_params->cohesion_coeff * cohesions[i]
			+ simulation_params->separation_coeff * separations[i]
			+ simulation_params->wall_separation_coeff * wall_separations[i];

		ssbo_velocities[i] = utils::math::normalize_or_zero(ssbo_velocities[i] + accel_blend * delta_time); //v = u + at
		ssbo_positions[i] += ssbo_velocities[i] * simulation_params->boid_speed * delta_time; //s = vt
		ssbo_positions[i] = clamp(ssbo_positions[i], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
	}

}

namespace utils::runners
{
	gpu_ssbo::gpu_ssbo(simulation_parameters params) :
		ssbo_runner{ {"shaders/ssbo.vert", "shaders/basic.frag"}, params },
		amount{ params.boid_amount },
		ssbo_positions_dptr{ nullptr },
		ssbo_velocities_dptr{ nullptr },
		block_size{ 128 },
		grid_size{ static_cast<size_t>(utils::math::ceil(amount, block_size)) }
	{
		setup_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 0, 0);
		setup_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 1, 0);

		ssbo_positions_dptr  = (float4*)cuda_gl_manager.add_resource(ssbo_positions , cudaGraphicsMapFlagsNone);
		ssbo_velocities_dptr = (float4*)cuda_gl_manager.add_resource(ssbo_velocities, cudaGraphicsMapFlagsNone);

		float spawn_range = sim_params.cube_size * 0.5f;
		utils::cuda::containers::random_vec4_fill_dptr(ssbo_positions_dptr, amount, -spawn_range, spawn_range);
		utils::cuda::containers::random_vec4_fill_dptr(ssbo_velocities_dptr, amount, -1, 1);

		// manual transfers are sufficient since transfers on these variables are occasional and one-sided only (CPU->GPU)
		cudaMalloc(&sim_params_dptr, sizeof(simulation_parameters) );
		cudaMemcpy(sim_params_dptr, &sim_params, sizeof(simulation_parameters), cudaMemcpyHostToDevice);

		cudaMalloc(&sim_volume_dptr, sizeof(utils::math::plane) * 6);
		cudaMemcpy(sim_volume_dptr, sim_volume.data(), sizeof(utils::math::plane) * 6, cudaMemcpyHostToDevice);
		
		// cudaMalloc is sufficient since no transfers between CPU-GPU (no cudaMemcpy)
		cudaMalloc(&alignments_dptr      , amount * sizeof(float4));
		cudaMalloc(&cohesions_dptr       , amount * sizeof(float4));
		cudaMalloc(&separations_dptr     , amount * sizeof(float4));
		cudaMalloc(&wall_separations_dptr, amount * sizeof(float4));

		// create streams
		cudaStreamCreate(&ali_stream); cudaStreamCreate(&coh_stream); cudaStreamCreate(&sep_stream); cudaStreamCreate(&wsp_stream);
	}

	void gpu_ssbo::naive_calculation(const float delta_time)
	{
		alignment       CUDA_KERNEL(grid_size, block_size, 0, ali_stream)(alignments_dptr, ssbo_positions_dptr, ssbo_velocities_dptr, amount, sim_params_dptr);
		cohesion        CUDA_KERNEL(grid_size, block_size, 0, coh_stream)(cohesions_dptr, ssbo_positions_dptr, amount, sim_params_dptr);
		separation      CUDA_KERNEL(grid_size, block_size, 0, sep_stream)(separations_dptr, ssbo_positions_dptr, amount, sim_params_dptr);
		wall_separation CUDA_KERNEL(grid_size, block_size, 0, wsp_stream)(wall_separations_dptr, ssbo_positions_dptr, sim_volume_dptr, amount);
		cudaDeviceSynchronize();

		blender CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, alignments_dptr, cohesions_dptr, separations_dptr, wall_separations_dptr, sim_params_dptr, amount, delta_time);
		cudaDeviceSynchronize();
	}

	void gpu_ssbo::calculate(const float delta_time)
	{
		naive_calculation(delta_time);
	}

	void gpu_ssbo::draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix)
	{
		// Update references to view and projection matrices
		update_buffer_object(ubo_matrices, GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), 1, (void*)glm::value_ptr(view_matrix));
		update_buffer_object(ubo_matrices, GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), 1, (void*)glm::value_ptr(projection_matrix));

		// Setup and draw debug info (simulation volume, ...)
		debug_shader.use();
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		cube_mesh.draw(GL_LINES);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		// Setup and draw boids
		boid_shader.use();
		triangle_mesh.draw_instanced(amount);
	}

	gpu_ssbo::~gpu_ssbo()
	{
		cudaStreamDestroy(ali_stream); cudaStreamDestroy(coh_stream); cudaStreamDestroy(sep_stream); cudaStreamDestroy(wsp_stream);
		cudaFree(&sim_params_dptr);
		cudaFree(&sim_volume_dptr);
		cudaFree(&alignments_dptr      );
		cudaFree(&cohesions_dptr       );
		cudaFree(&separations_dptr     );
		cudaFree(&wall_separations_dptr);
	}

	gpu_ssbo::simulation_parameters gpu_ssbo::get_simulation_parameters()
	{
		return sim_params;
	}

	void gpu_ssbo::set_simulation_parameters(simulation_parameters new_params)
	{
		sim_params = new_params;
		cudaMemcpy(sim_params_dptr, &sim_params, 1, cudaMemcpyHostToDevice);
		// TODO ricrea planes se cube_size è stato modificato nei parametri
	}
}