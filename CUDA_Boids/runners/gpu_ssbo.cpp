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

namespace utils::runners
{
	gpu_ssbo::gpu_ssbo(simulation_parameters params) :
		ssbo_runner{ {"shaders/ssbo.vert", "shaders/basic.frag"}, params },
		amount{ params.static_params.boid_amount },
		ssbo_positions_dptr{ nullptr },
		ssbo_velocities_dptr{ nullptr },
		block_size{ 256 },
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
		checkCudaErrors(cudaMalloc(&sim_params_dptr, sizeof(simulation_parameters)));
		checkCudaErrors(cudaMemcpy(sim_params_dptr, &sim_params, sizeof(simulation_parameters), cudaMemcpyHostToDevice));

		// The cube volume is constant so we can use cuda constant memory
		checkCudaErrors(cudaMemcpyToSymbol(sim_volume_cdptr, sim_volume.data(), sizeof(utils::math::plane) * 6, 0, cudaMemcpyHostToDevice));
		
		// cudaMalloc is sufficient since no transfers between CPU-GPU (no cudaMemcpy)
		checkCudaErrors(cudaMalloc(&alignments_dptr      , amount * sizeof(float4)));
		checkCudaErrors(cudaMalloc(&cohesions_dptr       , amount * sizeof(float4)));
		checkCudaErrors(cudaMalloc(&separations_dptr     , amount * sizeof(float4)));
		checkCudaErrors(cudaMalloc(&wall_separations_dptr, amount * sizeof(float4)));

		// create streams
		cudaStreamCreate(&ali_stream); cudaStreamCreate(&coh_stream); cudaStreamCreate(&sep_stream); cudaStreamCreate(&wsp_stream);

		// create arrays for grid calculations
		float boid_fov = sim_params.dynamic_params.boid_fov;
		float cell_size = 2 * boid_fov;
		grid_resolution = sim_params.static_params.cube_size / cell_size;
		int cell_amount = grid_resolution * grid_resolution * grid_resolution;

		checkCudaErrors(cudaMalloc(&boid_cell_indices_dptr, sizeof(behaviours::boid_cell_index) * amount));
		checkCudaErrors(cudaMalloc(&cell_idx_range_start_dptr, sizeof(int) * cell_amount));
		checkCudaErrors(cudaMalloc(&cell_idx_range_end_dptr  , sizeof(int) * cell_amount));

		cudaStreamCreate(&bci_stream); cudaStreamCreate(&cir_stream);

		// create arrays for swap
		checkCudaErrors(cudaMalloc(&sorted_positions_dptr   , sizeof(float4) * amount)); 
		checkCudaErrors(cudaMalloc(&sorted_velocities_dptr  , sizeof(float4) * amount));
		checkCudaErrors(cudaMalloc(&sorted_cell_indices_dptr, sizeof(int)    * amount));

		cudaStreamCreate(&pos_stream); cudaStreamCreate(&vel_stream);
	}

	void gpu_ssbo::naive_calculation(const float delta_time)
	{
		// Calculate velocities and apply velocity from all behaviours concurrently
		//behaviours::gpu::naive::flock CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, amount, sim_params.dynamic_params.boid_fov, sim_volume_cdptr, sim_params_dptr, delta_time);
		//cudaDeviceSynchronize();
	}

	// TODO maybe move helper methods to a prettier place instead of an anonymous namespace
	namespace
	{
		// all pointers are dptrs
		inline __global__ void assign_grid_indices(behaviours::boid_cell_index* boid_cell_indices, const float4* boid_positions, const size_t boid_amount, const float grid_extent, const float grid_resolution)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= boid_amount) return; // avoid threads ids overflowing the array

			int cell_amount = grid_resolution * grid_resolution * grid_resolution;
			float cube_half_size = grid_extent / 2;

			// trying to work with local/register memory as much as possible
			const float4 current_position = boid_positions[current];
			behaviours::boid_cell_index new_mapping;

			int x = utils::math::normalized_value_in_range(current_position.x, -cube_half_size, cube_half_size) * grid_resolution;
			int y = utils::math::normalized_value_in_range(current_position.y, -cube_half_size, cube_half_size) * grid_resolution;
			int z = utils::math::normalized_value_in_range(current_position.z, -cube_half_size, cube_half_size) * grid_resolution;
			int linear_index = x * grid_resolution * grid_resolution + y * grid_resolution + z;
			new_mapping.cell_id = clamp(linear_index, 0, cell_amount - 1);
			new_mapping.boid_id = current;

			boid_cell_indices[current] = new_mapping;
		}

		// TODO its better to split in two arrays (starts and ends...) for better and coalescent memory access 
		inline __global__ void find_cell_boid_range(int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr, const behaviours::boid_cell_index* boid_cell_indices, const size_t boid_amount)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= boid_amount) return; // avoid threads ids overflowing the array

			int current_boid_cell = boid_cell_indices[current].cell_id;
			
			// for last cell
			if (current == (boid_amount - 1))
			{
				cell_idx_range_end_dptr[current_boid_cell] = boid_amount;
				return;
			}
			
			int next_boid_cell = boid_cell_indices[current + 1].cell_id;

			// if cell changed
			if (current_boid_cell != next_boid_cell)
			{
				// value (current + 1) is because of internal convention, as range interval is considered [inclusive, exclusive)
				cell_idx_range_end_dptr[current_boid_cell] = current + 1;
				cell_idx_range_start_dptr[next_boid_cell]  = current + 1;
			}
		}
		
		struct order_by_cell_id
		{
			__host__ __device__ bool operator() (const behaviours::boid_cell_index& a, const behaviours::boid_cell_index& b) const
			{
				return a.cell_id < b.cell_id;
			}
		};

		// reorders the content of src into dst by using bci for the ordering
		inline __global__ void reorder_by_bci(float4* dst, const float4* src, const behaviours::boid_cell_index* boid_cell_indices, const size_t boid_amount)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= boid_amount) return; // avoid threads ids overflowing the array

			behaviours::boid_cell_index current_boid_cell = boid_cell_indices[current];

			dst[current] = src[current_boid_cell.boid_id];
		}
		inline __global__ void reorder_by_bci(int* dst, const behaviours::boid_cell_index* boid_cell_indices, const size_t boid_amount)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= boid_amount) return; // avoid threads ids overflowing the array

			behaviours::boid_cell_index current_boid_cell = boid_cell_indices[current];

			dst[current] = current_boid_cell.cell_id;
		}
	}
	
	void gpu_ssbo::uniform_grid_calculation(const float delta_time) 
	{
		//namespace bhvr = behaviours;
		//
		//int cell_amount = std::max(grid_resolution * grid_resolution * grid_resolution, 1.f);
		//
		////Reset the grid arrays
		//checkCudaErrors(cudaMemsetAsync(boid_cell_indices_dptr, 0, sizeof(behaviours::boid_cell_index) * amount, bci_stream));
		//checkCudaErrors(cudaMemsetAsync(cell_idx_range_dptr   , 0, sizeof(behaviours::idx_range) * cell_amount , cir_stream));
		//checkCudaErrors(cudaDeviceSynchronize());
		//
		//// Assign linear grid index to each boid
		//assign_grid_indices CUDA_KERNEL(grid_size, block_size)(boid_cell_indices_dptr, ssbo_positions_dptr, amount, sim_params.static_params.cube_size, grid_resolution);
		//cudaDeviceSynchronize();
		//
		//// Sort boids by grid index
		//thrust::device_ptr<bhvr::boid_cell_index> thrust_bci(boid_cell_indices_dptr);
		//thrust::sort(thrust_bci, thrust_bci + amount, order_by_cell_id());
		//
		//// Find ranges for boids living in the same grid cell
		//find_cell_boid_range CUDA_KERNEL(grid_size, block_size)(cell_idx_range_dptr, boid_cell_indices_dptr, amount);
		//cudaDeviceSynchronize();
		//
		//// Calculate velocities and apply velocity from all behaviours concurrently
		//bhvr::gpu::grid::uniform::flock CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, amount, boid_cell_indices_dptr, cell_idx_range_dptr, sim_params.dynamic_params.boid_fov, sim_volume_dptr, sim_params_dptr, delta_time);
		//cudaDeviceSynchronize();
	}

	void gpu_ssbo::coherent_grid_calculation(const float delta_time) 
	{
		namespace bhvr = behaviours;

		int cell_amount = std::max(grid_resolution * grid_resolution * grid_resolution, 1.f);

		//Reset the grid arrays
		checkCudaErrors(cudaMemsetAsync(boid_cell_indices_dptr, 0, sizeof(behaviours::boid_cell_index) * amount, bci_stream));
		checkCudaErrors(cudaMemsetAsync(cell_idx_range_start_dptr   , 0, sizeof(int) * cell_amount, ali_stream)); // TODO temporary stream (to rename/change)
		checkCudaErrors(cudaMemsetAsync(cell_idx_range_end_dptr     , 0, sizeof(int) * cell_amount, coh_stream)); // TODO temporary stream (to rename/change)
		checkCudaErrors(cudaDeviceSynchronize());

		// Assign linear grid index to each boid 
		assign_grid_indices CUDA_KERNEL(grid_size, block_size)(boid_cell_indices_dptr, ssbo_positions_dptr, amount, sim_params.static_params.cube_size, grid_resolution);
		
		// Sort boids by grid index (thrust uses default stream so implicit sync)
		thrust::device_ptr<bhvr::boid_cell_index> thrust_bci(boid_cell_indices_dptr);
		thrust::sort(thrust_bci, thrust_bci + amount, order_by_cell_id());

		// Reorder vel/pos in another array
		reorder_by_bci CUDA_KERNEL(grid_size, block_size, 0, pos_stream) (sorted_positions_dptr, ssbo_positions_dptr , boid_cell_indices_dptr, amount);
		reorder_by_bci CUDA_KERNEL(grid_size, block_size, 0, vel_stream) (sorted_velocities_dptr, ssbo_velocities_dptr, boid_cell_indices_dptr, amount);
		reorder_by_bci CUDA_KERNEL(grid_size, block_size, 0, bci_stream) (sorted_cell_indices_dptr, boid_cell_indices_dptr, amount);

		// Copy values back to ssbo arrays (we need this as to copy the sorted data into the gl-managed ssbo)
		//checkCudaErrors(cudaMemcpyAsync(ssbo_positions_dptr , positions_aux_dptr , sizeof(float4) * amount, cudaMemcpyDeviceToDevice, pos_stream));
		//checkCudaErrors(cudaMemcpyAsync(ssbo_velocities_dptr, velocities_aux_dptr, sizeof(float4) * amount, cudaMemcpyDeviceToDevice, vel_stream));

		// Find ranges for boids living in the same grid cell
		find_cell_boid_range CUDA_KERNEL(grid_size, block_size, 0, cir_stream)(cell_idx_range_start_dptr, cell_idx_range_end_dptr, boid_cell_indices_dptr, amount);
		//cudaDeviceSynchronize();

		// Calculate velocities and apply velocity from all behaviours concurrently
		bhvr::gpu::grid::coherent::flock CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr, sorted_positions_dptr, sorted_velocities_dptr, sorted_cell_indices_dptr, amount, cell_idx_range_start_dptr, cell_idx_range_end_dptr, sim_params.dynamic_params.boid_fov, sim_params_dptr, delta_time);
		//bhvr::gpu::grid::coherent::alignment  CUDA_KERNEL(grid_size, block_size, 0, ali_stream)(alignments_dptr,       sorted_positions_dptr, sorted_velocities_dptr, sorted_cell_indices_dptr, amount, cell_idx_range_dptr, sim_params.dynamic_params.boid_fov);
		//bhvr::gpu::grid::coherent::cohesion   CUDA_KERNEL(grid_size, block_size, 0, coh_stream)(cohesions_dptr,        sorted_positions_dptr, sorted_cell_indices_dptr, amount, cell_idx_range_dptr, sim_params.dynamic_params.boid_fov);
		//bhvr::gpu::grid::coherent::separation CUDA_KERNEL(grid_size, block_size, 0, sep_stream)(separations_dptr,      sorted_positions_dptr, sorted_cell_indices_dptr, amount, cell_idx_range_dptr, sim_params.dynamic_params.boid_fov);
		//bhvr::gpu::wall_separation            CUDA_KERNEL(grid_size, block_size, 0, wsp_stream)(wall_separations_dptr, sorted_positions_dptr, sim_volume_dptr, amount);
		//
		//bhvr::gpu::grid::coherent::blender CUDA_KERNEL(grid_size, block_size)(ssbo_positions_dptr, ssbo_velocities_dptr,
		//	positions_aux_dptr, velocities_aux_dptr,
		//	alignments_dptr, cohesions_dptr, separations_dptr, wall_separations_dptr, sim_params_dptr, amount, delta_time);
		//
		cudaDeviceSynchronize();
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
		checkCudaErrors(cudaStreamDestroy(ali_stream)); 
		checkCudaErrors(cudaStreamDestroy(coh_stream)); 
		checkCudaErrors(cudaStreamDestroy(sep_stream)); 
		checkCudaErrors(cudaStreamDestroy(wsp_stream));
		checkCudaErrors(cudaFree(sim_params_dptr));
		checkCudaErrors(cudaFree(alignments_dptr      ));
		checkCudaErrors(cudaFree(cohesions_dptr       ));
		checkCudaErrors(cudaFree(separations_dptr     ));
		checkCudaErrors(cudaFree(wall_separations_dptr));

		checkCudaErrors(cudaStreamDestroy(bci_stream)); 
		checkCudaErrors(cudaStreamDestroy(cir_stream));
		checkCudaErrors(cudaFree(boid_cell_indices_dptr));
		checkCudaErrors(cudaFree(cell_idx_range_start_dptr));
		checkCudaErrors(cudaFree(cell_idx_range_end_dptr));

		checkCudaErrors(cudaStreamDestroy(pos_stream)); 
		checkCudaErrors(cudaStreamDestroy(vel_stream));
		checkCudaErrors(cudaFree(sorted_positions_dptr));
		checkCudaErrors(cudaFree(sorted_velocities_dptr));
		checkCudaErrors(cudaFree(sorted_cell_indices_dptr));
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
			checkCudaErrors(cudaMemcpy(sim_params_dptr, &sim_params, sizeof(simulation_parameters), cudaMemcpyHostToDevice));
		}
	}
}