#include "gpu_ssbo.h"

// std libraries
#include <vector>
#include <math.h>

#include <glm/glm.hpp>

// CUDA libraries
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// utils libraries
#include "../utils/CUDA/vector_math.h"
#include "../utils/CUDA/cuda_utils.h"
#include "../utils/utils.h"

__constant__ utils::math::plane sim_volume_cdptr[6];
__constant__ utils::runners::boid_runner::simulation_parameters sim_params_cmem;

namespace utils::runners
{
	gpu_ssbo::gpu_ssbo(simulation_parameters params) :
		ssbo_runner{ params },
		amount{ params.static_params.boid_amount },
		ssbo_positions_dptr{ nullptr },
		ssbo_velocities_dptr{ nullptr },
		block_size{ 128 }, // block_size suggested by nvprof
		grid_size{ static_cast<size_t>(utils::math::ceil(amount, block_size)) }
	{
		setup_buffer_object(ssbo_positions , GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 0, 0);
		setup_buffer_object(ssbo_velocities, GL_SHADER_STORAGE_BUFFER, sizeof(float4), amount, 1, 0);

		ssbo_positions_dptr  = (float4*)cuda_gl_manager.add_resource(ssbo_positions , cudaGraphicsMapFlagsNone);
		ssbo_velocities_dptr = (float4*)cuda_gl_manager.add_resource(ssbo_velocities, cudaGraphicsMapFlagsNone);

		int spawn_range = static_cast<int>(sim_params.static_params.cube_size * 0.5f - 0.001f);
		utils::cuda::containers::random_vec4_fill_dptr(ssbo_positions_dptr, amount, -spawn_range, spawn_range);
		utils::cuda::containers::random_vec4_fill_dptr(ssbo_velocities_dptr, amount, -1, 1);

		// we are not using shared memory so we prefer to have a bigger l1 cache
		//checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

		// manual transfers are sufficient since transfers on these variables are occasional and one-sided only (CPU->GPU) and are constant while kernel runs
		checkCudaErrors(cudaMemcpyToSymbol(sim_params_cmem, &sim_params, sizeof(simulation_parameters)));

		// The cube volume is constant so we can use cuda constant memory
		checkCudaErrors(cudaMemcpyToSymbol(sim_volume_cdptr, sim_volume.data(), sizeof(utils::math::plane) * 6));
		
		//checkCudaErrors(cudaMalloc(&alignments_dptr      , amount * sizeof(float4)));
		//checkCudaErrors(cudaMalloc(&cohesions_dptr       , amount * sizeof(float4)));
		//checkCudaErrors(cudaMalloc(&separations_dptr     , amount * sizeof(float4)));
		//checkCudaErrors(cudaMalloc(&wall_separations_dptr, amount * sizeof(float4)));
		//
		//// create streams
		//cudaStreamCreate(&ali_stream); cudaStreamCreate(&coh_stream); cudaStreamCreate(&sep_stream); cudaStreamCreate(&wsp_stream);

		// create arrays for grid calculations
		int boid_fov = sim_params.dynamic_params.boid_fov;
		float cell_size = 2 * boid_fov;
		grid_resolution = sim_params.static_params.cube_size / cell_size;
		int cell_amount = static_cast<int>(grid_resolution * grid_resolution * grid_resolution);

		// cudaMalloc is sufficient since no transfers between CPU-GPU (no cudaMemcpy)
		checkCudaErrors(cudaMalloc(&bci_boid_indices_dptr, sizeof(int) * amount));
		checkCudaErrors(cudaMalloc(&bci_cell_indices_dptr, sizeof(int) * amount));
		checkCudaErrors(cudaMalloc(&cell_idx_range_start_dptr, sizeof(int) * cell_amount));
		checkCudaErrors(cudaMalloc(&cell_idx_range_end_dptr  , sizeof(int) * cell_amount));

		cudaStreamCreate(&bci_stream); cudaStreamCreate(&cir_stream); cudaStreamCreate(&cir_stream2);

		// create arrays for swap
		checkCudaErrors(cudaMalloc(&sorted_positions_dptr   , sizeof(float4) * amount)); 
		checkCudaErrors(cudaMalloc(&sorted_velocities_dptr  , sizeof(float4) * amount));

		cudaStreamCreate(&pos_stream); cudaStreamCreate(&vel_stream);
	}

	void gpu_ssbo::naive_calculation(const float delta_time)
	{
		// Calculate velocities and apply velocity from all behaviours concurrently
		behaviours::gpu::naive::flock CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, amount, sim_params.dynamic_params.boid_fov, delta_time);
		cudaDeviceSynchronize();
	}

	namespace
	{
		inline __global__ void assign_grid_indices(int* bci_boid_indices_dptr, int* bci_cell_indices_dptr, const float4* boid_positions, const size_t boid_amount, const float grid_extent, const float grid_resolution)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= boid_amount) return; // avoid threads ids overflowing the array

			int cell_amount = max(1.f, grid_resolution * grid_resolution * grid_resolution);
			float cube_half_size = grid_extent / 2;

			// trying to work with local/register memory as much as possible
			const float4 current_position = boid_positions[current];

			int x = utils::math::normalized_value_in_range(current_position.x, -cube_half_size, cube_half_size) * grid_resolution;
			int y = utils::math::normalized_value_in_range(current_position.y, -cube_half_size, cube_half_size) * grid_resolution;
			int z = utils::math::normalized_value_in_range(current_position.z, -cube_half_size, cube_half_size) * grid_resolution;
			int linear_index = x * grid_resolution * grid_resolution + y * grid_resolution + z;
			bci_cell_indices_dptr[current] = clamp(linear_index, 0, cell_amount - 1);
			bci_boid_indices_dptr[current] = current;
		}

		inline __global__ void find_cell_boid_range(int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr, const int* bci_cell_indices_dptr, const size_t boid_amount)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= boid_amount) return; // avoid threads ids overflowing the array

			int current_boid_cell = bci_cell_indices_dptr[current];

			// for last cell
			if (current == (boid_amount - 1))
			{
				cell_idx_range_end_dptr[current_boid_cell] = boid_amount;
				return;
			}

			int next_boid_cell = bci_cell_indices_dptr[current + 1];

			// if cell changed
			if (current_boid_cell != next_boid_cell)
			{
				// value (current + 1) is because of internal convention, as range interval is considered [inclusive, exclusive)
				cell_idx_range_end_dptr[current_boid_cell] = current + 1;
				cell_idx_range_start_dptr[next_boid_cell] = current + 1;
			}
		}

		inline __global__ void reorder_by_bci(float4* sorted_positions, const float4* positions,
			float4* sorted_velocities, const float4* velocities,
			int* bci_boid_indices_dptr, int* bci_cell_indices_dptr, const size_t boid_amount)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= boid_amount) return; // avoid threads ids overflowing the array

			int current_boid_id = bci_boid_indices_dptr[current];

			sorted_positions[current] = positions[current_boid_id];
			sorted_velocities[current] = velocities[current_boid_id];
		}
	}
	
	void gpu_ssbo::uniform_grid_calculation(const float delta_time) 
	{
		namespace bhvr = behaviours;
		
		int cell_amount = std::max(1.f, grid_resolution * grid_resolution * grid_resolution);
		
		//Reset the grid arrays
		checkCudaErrors(cudaMemsetAsync(cell_idx_range_start_dptr   , 0, sizeof(int) * cell_amount, cir_stream));
		checkCudaErrors(cudaMemsetAsync(cell_idx_range_end_dptr     , 0, sizeof(int) * cell_amount, cir_stream2));
		
		// Assign linear grid index to each boid
		assign_grid_indices CUDA_KERNEL(grid_size, block_size, 0, bci_stream)(bci_boid_indices_dptr, bci_cell_indices_dptr, ssbo_positions_dptr, amount, sim_params.static_params.cube_size, grid_resolution);
		
		// Use thrust to sort boids indices by grid index (thrust uses default stream so implicit sync)
		thrust::device_ptr<int> thrust_bci_cell_ids(bci_cell_indices_dptr);
		thrust::device_ptr<int> thrust_bci_boid_ids(bci_boid_indices_dptr);
		thrust::sort_by_key(thrust_bci_cell_ids, thrust_bci_cell_ids + amount, thrust_bci_boid_ids);
		
		// Find ranges for boids living in the same grid cell
		find_cell_boid_range CUDA_KERNEL(grid_size, block_size, 0, cir_stream)(cell_idx_range_start_dptr, cell_idx_range_end_dptr, bci_cell_indices_dptr, amount);

		// Calculate velocities and apply velocity from all behaviours concurrently
		bhvr::gpu::grid::uniform::flock CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, amount, bci_boid_indices_dptr, bci_cell_indices_dptr, cell_idx_range_start_dptr, cell_idx_range_end_dptr, sim_params.dynamic_params.boid_fov, delta_time);
		
		cudaDeviceSynchronize(); // We wait for the calculations to finish before feeding the info to OpenGL
	}

	void gpu_ssbo::coherent_grid_calculation(const float delta_time) 
	{
		namespace bhvr = behaviours;

		int cell_amount = std::max(1.f, grid_resolution * grid_resolution * grid_resolution);

		//Reset the grid arrays
		checkCudaErrors(cudaMemsetAsync(cell_idx_range_start_dptr   , 0, sizeof(int) * cell_amount, cir_stream));
		checkCudaErrors(cudaMemsetAsync(cell_idx_range_end_dptr     , 0, sizeof(int) * cell_amount, cir_stream2)); 

		// Assign linear grid index to each boid 
		assign_grid_indices CUDA_KERNEL(grid_size, block_size, 0, bci_stream)(bci_boid_indices_dptr, bci_cell_indices_dptr, ssbo_positions_dptr, amount, sim_params.static_params.cube_size, grid_resolution);
		
		// Use thrust to sort boids indices by grid index (thrust uses default stream so implicit sync)
		thrust::device_ptr<int> thrust_bci_cell_ids(bci_cell_indices_dptr);
		thrust::device_ptr<int> thrust_bci_boid_ids(bci_boid_indices_dptr);
		thrust::sort_by_key(thrust_bci_cell_ids, thrust_bci_cell_ids + amount, thrust_bci_boid_ids);

		// Reorder vel/pos in another array
		reorder_by_bci CUDA_KERNEL(grid_size, block_size, 0, bci_stream) (sorted_positions_dptr, ssbo_positions_dptr, sorted_velocities_dptr, ssbo_velocities_dptr, bci_boid_indices_dptr, bci_cell_indices_dptr, amount);
		
		// Find ranges for boids living in the same grid cell
		find_cell_boid_range CUDA_KERNEL(grid_size, block_size, 0, cir_stream)(cell_idx_range_start_dptr, cell_idx_range_end_dptr, bci_cell_indices_dptr, amount);

		// Calculate velocities and apply velocity from all behaviours concurrently
		bhvr::gpu::grid::coherent::flock CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, sorted_positions_dptr, sorted_velocities_dptr, bci_cell_indices_dptr, amount, cell_idx_range_start_dptr, cell_idx_range_end_dptr, sim_params.dynamic_params.boid_fov, delta_time);
		
		cudaDeviceSynchronize(); // We wait for the calculations to finish before feeding the info to OpenGL
	}

	// TODO see if you can make this template and not check every loop the condition (sim_type aint gonna change during runtime) (https://stackoverflow.com/questions/56742898/avoid-checking-the-same-condition-every-step-in-a-loop-in-c)
	void gpu_ssbo::calculate(const float delta_time)
	{
		switch (sim_params.static_params.sim_type)
		{
		case NAIVE:
			naive_calculation(delta_time);
			break;
		case UNIFORM_GRID:
			uniform_grid_calculation(delta_time);
			break;
		case COHERENT_GRID:
			coherent_grid_calculation(delta_time);
			break;
		}
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
		//checkCudaErrors(cudaStreamDestroy(ali_stream)); 
		//checkCudaErrors(cudaStreamDestroy(coh_stream)); 
		//checkCudaErrors(cudaStreamDestroy(sep_stream)); 
		//checkCudaErrors(cudaStreamDestroy(wsp_stream));
		//checkCudaErrors(cudaFree(alignments_dptr      ));
		//checkCudaErrors(cudaFree(cohesions_dptr       ));
		//checkCudaErrors(cudaFree(separations_dptr     ));
		//checkCudaErrors(cudaFree(wall_separations_dptr));

		checkCudaErrors(cudaStreamDestroy(bci_stream)); 
		checkCudaErrors(cudaStreamDestroy(cir_stream));
		checkCudaErrors(cudaStreamDestroy(cir_stream2));
		checkCudaErrors(cudaFree(bci_boid_indices_dptr));
		checkCudaErrors(cudaFree(bci_cell_indices_dptr));
		checkCudaErrors(cudaFree(cell_idx_range_start_dptr));
		checkCudaErrors(cudaFree(cell_idx_range_end_dptr));

		checkCudaErrors(cudaStreamDestroy(pos_stream)); 
		checkCudaErrors(cudaStreamDestroy(vel_stream));
		checkCudaErrors(cudaFree(sorted_positions_dptr));
		checkCudaErrors(cudaFree(sorted_velocities_dptr));
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
			// If the boid_fov parameter changed, the grid must be rebuilt 
			if (sim_params.dynamic_params.boid_fov != new_dyn_params.boid_fov)
			{
				checkCudaErrors(cudaFree(cell_idx_range_start_dptr));
				checkCudaErrors(cudaFree(cell_idx_range_end_dptr));

				float boid_fov = new_dyn_params.boid_fov;
				float cell_size = 2 * boid_fov;
				grid_resolution = sim_params.static_params.cube_size / cell_size;

				int cell_amount = std::max(grid_resolution * grid_resolution * grid_resolution, 1.f);
				checkCudaErrors(cudaMalloc(&cell_idx_range_start_dptr, sizeof(int) * cell_amount));
				checkCudaErrors(cudaMalloc(&cell_idx_range_end_dptr  , sizeof(int) * cell_amount));
			}
			// Update parameters (both on cpu and gpu)
			sim_params.dynamic_params = new_dyn_params;
			checkCudaErrors(cudaMemcpyToSymbol(sim_params_cmem, &sim_params, sizeof(simulation_parameters)));
		}
	}
}