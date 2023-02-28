#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "behaviour_utils.h"
#include "boid_runner.h"

extern __constant__ utils::math::plane sim_volume_cdptr[6];
extern __constant__ utils::runners::boid_runner::simulation_parameters sim_params_cmem;

#pragma region monolithic_behaviour
namespace utils::runners::behaviours::gpu
{
	namespace naive
	{
		inline __global__ void flock(float4* ssbo_positions, float4* ssbo_velocities,
			const float delta_time)
		{
			size_t current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= sim_params_cmem.static_params.boid_amount) return;

			float4 alignment{ 0 }, cohesion{ 0 }, separation{ 0 }, wall_separation{ 0 };
			float counter{ 0 };
			float4 repulsion{ 0 }; 
			float wall_distance{ 0 };

			bool in_radius, near_wall;
			float chs = sim_params_cmem.static_params.cube_size / 2; //chs -> cube half size
			float wall_repel_distance = 1 + (chs * 2) * .01f;

			float4 position_current = ssbo_positions[current], position_i;
			float4 new_position, new_velocity;

			for (size_t i = 0; i < sim_params_cmem.static_params.boid_amount; i++)
			{
				position_i = ssbo_positions[i];
				in_radius = utils::math::distance2(position_current, position_i) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;

				// alignment
				alignment += ssbo_velocities[i] * in_radius;

				// cohesion
				cohesion += position_i * in_radius;
				counter += 1.f * in_radius;

				// separation
				repulsion = position_current - position_i;
				separation += (repulsion / (utils::math::length2(repulsion) + 0.0001f)) * in_radius;

				// wall_separation
				for (size_t b = 0; b < 6; b++)
				{
					wall_distance = utils::math::distance_point_plane(ssbo_positions[current], sim_volume_cdptr[b]) + 0.0001f;
					near_wall = wall_distance < wall_repel_distance;
					wall_separation += (sim_volume_cdptr[b].normal / abs(wall_distance)) * near_wall;
				}
			}
			cohesion /= counter;
			cohesion -= position_current;

			float4 accel_blend = sim_params_cmem.dynamic_params.alignment_coeff * utils::math::normalize_or_zero(alignment)
				+ sim_params_cmem.dynamic_params.cohesion_coeff * utils::math::normalize_or_zero(cohesion)
				+ sim_params_cmem.dynamic_params.separation_coeff * utils::math::normalize_or_zero(separation)
				+ sim_params_cmem.dynamic_params.wall_separation_coeff * utils::math::normalize_or_zero(wall_separation);

			new_velocity = utils::math::normalize_or_zero(ssbo_velocities[current] + accel_blend * delta_time); //v = u + at
			new_position = clamp((position_current + new_velocity * sim_params_cmem.dynamic_params.boid_speed * delta_time), { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube

			ssbo_velocities[current] = new_velocity;
			ssbo_positions[current] = new_position;
		}
	}
	namespace grid
	{
		namespace uniform
		{
			inline __global__ void flock(float4* ssbo_positions, float4* ssbo_velocities,
				int* bci_boid_indices_dptr, int* bci_cell_indices_dptr,
				int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr,
				const float delta_time)
			{
				size_t current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= sim_params_cmem.static_params.boid_amount) return;

				float4 alignment{ 0 }, cohesion{ 0 }, separation{ 0 }, wall_separation{ 0 };
				float counter{ 0 };
				float4 repulsion{ 0 };
				float wall_distance{ 0 };

				int current_neighbour;
				int current_boid_id      = bci_boid_indices_dptr[current];
				int current_boid_cell_id = bci_cell_indices_dptr[current];
				int range_start = cell_idx_range_start_dptr[current_boid_cell_id], range_end = cell_idx_range_end_dptr[current_boid_cell_id];

				bool in_radius, near_wall;
				float chs = sim_params_cmem.static_params.cube_size / 2; //chs -> cube half size
				float wall_repel_distance = 1 + (chs * 2) * .01f;

				float4 position_current = ssbo_positions[current_boid_id], position_i;
				float4 new_position, new_velocity;

				for (size_t i = range_start; i < range_end; i++)
				{
					current_neighbour = bci_boid_indices_dptr[i];
					position_i = ssbo_positions[current_neighbour];
					in_radius = utils::math::distance2(position_current, position_i) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;

					// alignment
					alignment += ssbo_velocities[current_neighbour] * in_radius;

					// cohesion
					cohesion += position_i * in_radius; // baricenter buildup
					counter += 1.f * in_radius;

					// separation
					repulsion = position_current - position_i;
					separation += (repulsion / (utils::math::length2(repulsion) + 0.0001f)) * in_radius;

					// wall_separation
					for (size_t b = 0; b < 6; b++)
					{
						wall_distance = utils::math::distance_point_plane(ssbo_positions[current_boid_id], sim_volume_cdptr[b]) + 0.0001f;
						near_wall = wall_distance < wall_repel_distance;
						wall_separation += (sim_volume_cdptr[b].normal / abs(wall_distance)) * near_wall;
					}
				}
				cohesion /= counter;
				cohesion -= position_current;

				float4 accel_blend = sim_params_cmem.dynamic_params.alignment_coeff * utils::math::normalize_or_zero(alignment)
					+ sim_params_cmem.dynamic_params.cohesion_coeff * utils::math::normalize_or_zero(cohesion)
					+ sim_params_cmem.dynamic_params.separation_coeff * utils::math::normalize_or_zero(separation)
					+ sim_params_cmem.dynamic_params.wall_separation_coeff * utils::math::normalize_or_zero(wall_separation);

				new_velocity = utils::math::normalize_or_zero(ssbo_velocities[current_boid_id] + accel_blend * delta_time); //v = u + at
				new_position = clamp((position_current + new_velocity * sim_params_cmem.dynamic_params.boid_speed * delta_time), { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube

				ssbo_velocities[current_boid_id] = new_velocity;
				ssbo_positions [current_boid_id] = new_position;
			}
		}

		namespace coherent
		{
			// Single master behaviour including alignment, cohesion, separation and wall separation in one go: 
			// not modular but more efficient than separate behaviours because requires less global memory loads
			inline __global__ void flock(float4* ssbo_positions, float4* ssbo_velocities,
				float4* sorted_positions, float4* sorted_velocities, int* cell_ids,
				int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr, 
				const float delta_time)
			{
				size_t current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= sim_params_cmem.static_params.boid_amount) return;
				
				float4 alignment{ 0 }, cohesion{ 0 }, separation{ 0 }, wall_separation{ 0 };
				float counter{ 0 };
				float4 repulsion{ 0 };
				float wall_distance{ 0 };

				int boid_cell = cell_ids[current];
				int range_start = cell_idx_range_start_dptr[boid_cell], range_end = cell_idx_range_end_dptr[boid_cell];
				bool in_radius, near_wall;
				float chs = sim_params_cmem.static_params.cube_size / 2; //chs -> cube half size
				float wall_repel_distance = 1 + (chs * 2) * .01f;

				float4 position_current = sorted_positions[current], position_i;
				float4 new_position, new_velocity;

				for (size_t i = range_start; i < range_end; i++)
				{
					position_i = sorted_positions[i];
					in_radius = utils::math::distance2(position_current, position_i) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;

					// alignment
					alignment += sorted_velocities[i] * in_radius;

					// cohesion
					cohesion += position_i * in_radius; // baricenter buildup
					counter += 1.f * in_radius;

					// separation
					repulsion = position_current - position_i;
					separation += (repulsion / (utils::math::length2(repulsion) + 0.0001f)) * in_radius;

					// wall_separation
					for (size_t b = 0; b < 6; b++)
					{
						wall_distance = utils::math::distance_point_plane(position_current, sim_volume_cdptr[b]) + 0.0001f;
						near_wall = wall_distance < wall_repel_distance;
						wall_separation += (sim_volume_cdptr[b].normal / abs(wall_distance)) * near_wall; //wall repulsion calculation
					}
				}
				cohesion /= counter; // baricenter calculated
				cohesion -= position_current; // direction towards baricenter

				float4 accel_blend = sim_params_cmem.dynamic_params.alignment_coeff * utils::math::normalize_or_zero(alignment)
					+ sim_params_cmem.dynamic_params.cohesion_coeff * utils::math::normalize_or_zero(cohesion)
					+ sim_params_cmem.dynamic_params.separation_coeff * utils::math::normalize_or_zero(separation)
					+ sim_params_cmem.dynamic_params.wall_separation_coeff * utils::math::normalize_or_zero(wall_separation);

				new_velocity = utils::math::normalize_or_zero(sorted_velocities[current] + accel_blend * delta_time); //v = u + at
				new_position = clamp((position_current + new_velocity * sim_params_cmem.dynamic_params.boid_speed * delta_time), { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube

				ssbo_velocities[current] = new_velocity;
				ssbo_positions[current] = new_position;
			}
		}
	}
}
#pragma endregion

#pragma region modular_behaviour
namespace utils::runners::behaviours::gpu
{
	inline __global__ void wall_separation(float4* wall_separations, float4* positions)
	{
		int current = blockIdx.x * blockDim.x + threadIdx.x;
		if (current >= sim_params_cmem.static_params.boid_amount) return;

		float4 separation{ 0 };
		float4 repulsion;
		float distance;
		float near_wall;
		// wall check
		for (size_t b = 0; b < 6; b++)
		{
			distance = utils::math::distance_point_plane(positions[current], sim_volume_cdptr[b]) + 0.0001f;
			near_wall = distance < 1.f;
			repulsion = (sim_volume_cdptr[b].normal / abs(distance)) * near_wall;
			separation += repulsion;
		}

		wall_separations[current] = utils::math::normalize_or_zero(separation);
	}

	inline __global__ void blender(float4* ssbo_positions, float4* ssbo_velocities,
		float4* alignments, float4* cohesions, float4* separations, float4* wall_separations,
		const float delta_time)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= sim_params_cmem.static_params.boid_amount) return;

		float chs = sim_params_cmem.static_params.cube_size / 2;
		float4 accel_blend;

		accel_blend = sim_params_cmem.dynamic_params.alignment_coeff * alignments[i]
			+ sim_params_cmem.dynamic_params.cohesion_coeff * cohesions[i]
			+ sim_params_cmem.dynamic_params.separation_coeff * separations[i]
			+ sim_params_cmem.dynamic_params.wall_separation_coeff * wall_separations[i];

		ssbo_velocities[i] = utils::math::normalize_or_zero(ssbo_velocities[i] + accel_blend * delta_time); //v = u + at
		ssbo_positions[i] += ssbo_velocities[i] * sim_params_cmem.dynamic_params.boid_speed * delta_time; //s = vt
		ssbo_positions[i] = clamp(ssbo_positions[i], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
	}

	namespace naive
	{
		inline __global__ void alignment(float4* alignments, float4* positions, float4* velocities)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= sim_params_cmem.static_params.boid_amount) return; // avoid threads ids overflowing the array

			float4 alignment{ 0 };
			bool in_radius;
			for (size_t i = 0; i < sim_params_cmem.static_params.boid_amount; i++)
			{
				// condition as multiplier avoids warp divergence
				in_radius = utils::math::distance2(positions[current], positions[i]) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;
				alignment += velocities[i] * in_radius;
			}

			alignments[current] = utils::math::normalize_or_zero(alignment);

		}

		inline __global__ void cohesion(float4* cohesions, float4* positions)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= sim_params_cmem.static_params.boid_amount) return; // avoid threads ids overflowing the array

			float4 cohesion{ 0 }, baricenter{ 0 };
			float counter{ 0 };
			bool in_radius;
			for (size_t i = 0; i < sim_params_cmem.static_params.boid_amount; i++)
			{
				in_radius = utils::math::distance2(positions[current], positions[i]) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;

				baricenter += positions[i] * in_radius;
				counter += 1.f * in_radius;
			}
			baricenter /= counter;
			cohesion = baricenter - positions[current];
			cohesions[current] = utils::math::normalize_or_zero(cohesion);

		}

		inline __global__ void separation(float4* separations, float4* positions)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= sim_params_cmem.static_params.boid_amount) return; // avoid threads ids overflowing the array

			float4 separation{ 0 };
			float4 repulsion; float repulsion_length2;
			bool in_radius;

			// boid check
			for (size_t i = 0; i < sim_params_cmem.static_params.boid_amount; i++)
			{
				repulsion = positions[current] - positions[i];
				repulsion_length2 = utils::math::length2(repulsion);
				in_radius = repulsion_length2 < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;
				separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;
				separation += (utils::math::normalize_or_zero(repulsion) / (length(repulsion) + 0.0001f)) * in_radius; 
			}

			separations[current] = utils::math::normalize_or_zero(separation);
		}
	}

	namespace grid
	{
		namespace uniform
		{
			inline __global__ void alignment(float4* alignments, float4* positions, float4* velocities,
				int* bci_boid_indices_dptr, int* bci_cell_indices_dptr,
				int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= sim_params_cmem.static_params.boid_amount) return; // avoid threads ids overflowing the array
				
				float4 alignment{ 0 };

				int current_neighbour;
				int current_boid_id = bci_boid_indices_dptr[current];
				int current_boid_cell_id = bci_cell_indices_dptr[current];
				int range_start = cell_idx_range_start_dptr[current_boid_cell_id], range_end = cell_idx_range_end_dptr[current_boid_cell_id];

				bool in_radius;
				for (size_t i = range_start; i < range_end; i++)
				{
					current_neighbour = bci_boid_indices_dptr[i];
					in_radius = utils::math::distance2(positions[current_boid_id], positions[current_neighbour]) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;
					// condition as multiplier avoids warp divergence
					alignment += velocities[current_neighbour] * in_radius;
				}

				alignments[current_boid_id] = utils::math::normalize_or_zero(alignment);

			}

			inline __global__ void cohesion(float4* cohesions, float4* positions,
				int* bci_boid_indices_dptr, int* bci_cell_indices_dptr,
				int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= sim_params_cmem.static_params.boid_amount) return;

				float4 cohesion{ 0 }, baricenter{ 0 };
				float counter{ 0 };

				int current_neighbour;
				int current_boid_id = bci_boid_indices_dptr[current];
				int current_boid_cell_id = bci_cell_indices_dptr[current];
				int range_start = cell_idx_range_start_dptr[current_boid_cell_id], range_end = cell_idx_range_end_dptr[current_boid_cell_id];
				
				bool in_radius;
				for (size_t i = range_start; i < range_end; i++)
				{
					current_neighbour = bci_boid_indices_dptr[i];
					in_radius = utils::math::distance2(positions[current_boid_id], positions[current_neighbour]) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;

					baricenter += positions[current_neighbour] * in_radius;
					counter += 1.f * in_radius;
				}
				baricenter /= counter;
				cohesion = baricenter - positions[current_boid_id];
				cohesions[current_boid_id] = utils::math::normalize_or_zero(cohesion);

			}

			inline __global__ void separation(float4* separations, float4* positions,
				int* bci_boid_indices_dptr, int* bci_cell_indices_dptr,
				int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= sim_params_cmem.static_params.boid_amount) return;

				float4 separation{ 0 };
				float4 repulsion; float repulsion_length2;
				bool in_radius;

				int current_neighbour;
				int current_boid_id = bci_boid_indices_dptr[current];
				int current_boid_cell_id = bci_cell_indices_dptr[current];
				int range_start = cell_idx_range_start_dptr[current_boid_cell_id], range_end = cell_idx_range_end_dptr[current_boid_cell_id];

				for (size_t i = range_start; i < range_end; i++)
				{
					current_neighbour = bci_boid_indices_dptr[i];
					repulsion = positions[current_boid_id] - positions[current_neighbour];
					repulsion_length2 = utils::math::length2(repulsion);
					in_radius = repulsion_length2 < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;
					separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;
				}

				separations[current_boid_id] = utils::math::normalize_or_zero(separation);
			}
		}

		namespace coherent
		{
			inline __global__ void alignment(float4* alignments, float4* positions, float4* velocities, 
				int* cell_ids, int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= sim_params_cmem.static_params.boid_amount) return; // avoid threads ids overflowing the array
			
				float4 alignment{ 0 };
				int boid_cell = cell_ids[current];
				int range_start = cell_idx_range_start_dptr[boid_cell], range_end = cell_idx_range_end_dptr[boid_cell];
				bool in_radius;

				for (size_t i = range_start; i < range_end; i++)
				{
					in_radius = utils::math::distance2(positions[current], positions[i]) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;
					// condition as multiplier avoids warp divergence
					alignment += velocities[i] * in_radius;
				}
			
				alignments[current] = utils::math::normalize_or_zero(alignment);
				
			}

			inline __global__ void cohesion(float4* cohesions, float4* positions, 
				int* cell_ids, int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= sim_params_cmem.static_params.boid_amount) return; // avoid threads ids overflowing the array

				float4 cohesion{ 0 }, baricenter{ 0 };
				float counter{ 0 };
				int boid_cell = cell_ids[current];
				int range_start = cell_idx_range_start_dptr[boid_cell], range_end = cell_idx_range_end_dptr[boid_cell];
				bool in_radius;

				for (size_t i = range_start; i < range_end; i++)
				{
					in_radius = utils::math::distance2(positions[current], positions[i]) < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;

					baricenter += positions[i] * in_radius;
					counter += 1.f * in_radius;
				}
				baricenter /= counter;
				cohesion = baricenter - positions[current];

				cohesions[current] = utils::math::normalize_or_zero(cohesion);
			}

			inline __global__ void separation(float4* separations, float4* positions, 
				int* cell_ids, int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= sim_params_cmem.static_params.boid_amount) return; // avoid threads ids overflowing the array

				float4 separation{ 0 };
				float4 repulsion; float repulsion_length2;
				bool in_radius;

				int boid_cell = cell_ids[current];
				int range_start = cell_idx_range_start_dptr[boid_cell], range_end = cell_idx_range_end_dptr[boid_cell];

				for (size_t i = range_start; i < range_end; i++)
				{
					repulsion = positions[current] - positions[i];
					repulsion_length2 = utils::math::length2(repulsion); 
					in_radius = repulsion_length2 < sim_params_cmem.dynamic_params.boid_fov * sim_params_cmem.dynamic_params.boid_fov;
					separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;
				}

				separations[current] = utils::math::normalize_or_zero(separation);
			}

			
			// sorted positions and sorted velocities update the ssbo array without the need of copying stuff to those array first, saving time
			inline __global__ void blender(float4* ssbo_positions, float4* ssbo_velocities,
				float4* sorted_positions, float4* sorted_velocities,
				float4* alignments, float4* cohesions, float4* separations, float4* wall_separations,
				const float delta_time)
			{
				int i = blockIdx.x * blockDim.x + threadIdx.x;
				if (i >= sim_params_cmem.static_params.boid_amount) return; // avoid threads ids overflowing the array

				float chs = sim_params_cmem.static_params.cube_size / 2;
				float4 accel_blend, new_position = sorted_positions[i], new_velocity;

				accel_blend = sim_params_cmem.dynamic_params.alignment_coeff * alignments[i]
					+ sim_params_cmem.dynamic_params.cohesion_coeff * cohesions[i]
					+ sim_params_cmem.dynamic_params.separation_coeff * separations[i]
					+ sim_params_cmem.dynamic_params.wall_separation_coeff * wall_separations[i];

				new_velocity = utils::math::normalize_or_zero(sorted_velocities[i] + accel_blend * delta_time); //v = u + at
				new_position += new_velocity * sim_params_cmem.dynamic_params.boid_speed * delta_time; //s = vt
				new_position = clamp(new_position, { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
			
				ssbo_velocities[i] = new_velocity;
				ssbo_positions [i] = new_position;
			}
		}
	}
}
#pragma endregion
