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
#include "gpu_boid_behaviours.h"

namespace utils::runners
{
	gpu_ssbo::gpu_ssbo(simulation_parameters params) :
		ssbo_runner{ {"shaders/ssbo.vert", "shaders/basic.frag"}, params },
		amount{ params.static_params.boid_amount },
		ssbo_positions_dptr{ nullptr },
		ssbo_velocities_dptr{ nullptr },
		block_size{ 128 },
		grid_size{ static_cast<size_t>(utils::math::ceil(amount, block_size)) }
	{
		setup_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 0, 0);
		setup_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 1, 0);

		ssbo_positions_dptr  = (float4*)cuda_gl_manager.add_resource(ssbo_positions , cudaGraphicsMapFlagsNone);
		ssbo_velocities_dptr = (float4*)cuda_gl_manager.add_resource(ssbo_velocities, cudaGraphicsMapFlagsNone);

		float spawn_range = sim_params.static_params.cube_size * 0.5f;
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
		namespace n_bhvr = behaviours::naive::gpu;
		n_bhvr::alignment       CUDA_KERNEL(grid_size, block_size, 0, ali_stream)(alignments_dptr, ssbo_positions_dptr, ssbo_velocities_dptr, amount, sim_params.dynamic_params.boid_fov);
		n_bhvr::cohesion        CUDA_KERNEL(grid_size, block_size, 0, coh_stream)(cohesions_dptr, ssbo_positions_dptr, amount, sim_params.dynamic_params.boid_fov);
		n_bhvr::separation      CUDA_KERNEL(grid_size, block_size, 0, sep_stream)(separations_dptr, ssbo_positions_dptr, amount, sim_params.dynamic_params.boid_fov);
		n_bhvr::wall_separation CUDA_KERNEL(grid_size, block_size, 0, wsp_stream)(wall_separations_dptr, ssbo_positions_dptr, sim_volume_dptr, amount);
		cudaDeviceSynchronize();

		n_bhvr::blender CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, alignments_dptr, cohesions_dptr, separations_dptr, wall_separations_dptr, sim_params_dptr, amount, delta_time);
		cudaDeviceSynchronize();
	}

	// TODO bring helper methods from cpu

	void gpu_ssbo::uniform_grid_calculation(const float delta_time) 
	{
		// TODO
	}
	void gpu_ssbo::coherent_grid_calculation(const float delta_time) 
	{
		// TODO
	}

	void gpu_ssbo::calculate(const float delta_time)
	{
		naive_calculation(delta_time);
		//uniform_grid_calculation(delta_time);
		//coherent_grid_calculation(delta_time);
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

	void gpu_ssbo::set_dynamic_simulation_parameters(simulation_parameters::dynamic_parameters new_dyn_params)
	{
		bool params_changed = sim_params.dynamic_params != new_dyn_params;
		if (params_changed)
		{
			sim_params.dynamic_params = new_dyn_params;
			cudaMemcpy(sim_params_dptr, &sim_params, sizeof(simulation_parameters), cudaMemcpyHostToDevice);
			// TODO ricrea planes se cube_size è stato modificato nei parametri (solo se lo vuoi dinamico)
		}
	}
}