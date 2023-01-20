#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "behaviour_utils.h"
#include "boid_runner.h"

extern __constant__ utils::math::plane sim_volume_cdptr[6];

namespace utils::runners::behaviours::gpu
{
	inline __global__ void wall_separation(float4* wall_separations, float4* positions, utils::math::plane* borders, size_t amount)
	{
		int current = blockIdx.x * blockDim.x + threadIdx.x;
		if (current >= amount) return;

		float4 separation{ 0 };
		float4 repulsion;
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

	inline __global__ void blender(float4* ssbo_positions, float4* ssbo_velocities,
		float4* alignments, float4* cohesions, float4* separations, float4* wall_separations,
		utils::runners::boid_runner::simulation_parameters* simulation_params, size_t amount, const float delta_time)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= amount) return;

		float chs = simulation_params->static_params.cube_size / 2;
		float4 accel_blend;

		accel_blend = simulation_params->dynamic_params.alignment_coeff * alignments[i]
			+ simulation_params->dynamic_params.cohesion_coeff * cohesions[i]
			+ simulation_params->dynamic_params.separation_coeff * separations[i]
			+ simulation_params->dynamic_params.wall_separation_coeff * wall_separations[i];

		ssbo_velocities[i] = utils::math::normalize_or_zero(ssbo_velocities[i] + accel_blend * delta_time); //v = u + at
		ssbo_positions[i] += ssbo_velocities[i] * simulation_params->dynamic_params.boid_speed * delta_time; //s = vt
		ssbo_positions[i] = clamp(ssbo_positions[i], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
	}

	namespace naive
	{
		inline __global__ void alignment(float4* alignments, float4* positions, float4* velocities, size_t amount, size_t max_radius)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= amount) return; // avoid threads ids overflowing the array

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

		inline __global__ void cohesion(float4* cohesions, float4* positions, size_t amount, size_t max_radius)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= amount) return;

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

		inline __global__ void separation(float4* separations, float4* positions, size_t amount, size_t max_radius)
		{
			int current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= amount) return;

			float4 separation{ 0 };
			float4 repulsion; float repulsion_length2;
			bool in_radius;

			// boid check
			for (size_t i = 0; i < amount; i++)
			{
				repulsion = positions[current] - positions[i];
				repulsion_length2 = utils::math::length2(repulsion);
				in_radius = repulsion_length2 < max_radius* max_radius;
				separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;
				separation += (utils::math::normalize_or_zero(repulsion) / (length(repulsion) + 0.0001f)) * in_radius; 
			}

			separations[current] = utils::math::normalize_or_zero(separation);
		}
	
		inline __global__ void flock(float4* ssbo_positions, float4* ssbo_velocities, 
			size_t amount, size_t max_radius,
			utils::math::plane* borders,
			utils::runners::boid_runner::simulation_parameters* simulation_params, const float delta_time)
		{
			size_t current = blockIdx.x * blockDim.x + threadIdx.x;
			if (current >= amount) return;

			float4 alignment{ 0 }, cohesion{ 0 }, separation{ 0 }, wall_separation{ 0 };
			float4 baricenter{ 0 };
			float counter{ 0 };
			float4 repulsion{ 0 }; float repulsion_length2{ 0 };
			float4 wall_repulsion{ 0 }; float wall_distance{ 0 };

			bool in_radius, near_wall;
			float chs = simulation_params->static_params.cube_size / 2; //chs -> cube half size
			float wall_repel_distance = 1 + (chs * 2) * .01f;

			for (size_t i = 0; i < amount; i++)
			{
				in_radius = utils::math::distance2(ssbo_positions[current], ssbo_positions[i]) < max_radius * max_radius;

				// alignment
				alignment += ssbo_velocities[i] * in_radius;

				// cohesion
				baricenter += ssbo_positions[i] * in_radius;
				counter += 1.f * in_radius;

				// separation
				repulsion = ssbo_positions[current] - ssbo_positions[i];
				repulsion_length2 = utils::math::length2(repulsion);
				separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;

				// wall_separation
				for (size_t b = 0; b < 6; b++)
				{
					wall_distance = utils::math::distance_point_plane(ssbo_positions[current], borders[b]) + 0.0001f;
					near_wall = wall_distance < wall_repel_distance;
					wall_repulsion = (borders[b].normal / abs(wall_distance)) * near_wall;
					wall_separation += wall_repulsion;
				}
			}
			baricenter /= counter;
			cohesion = baricenter - ssbo_positions[current];

			float4 accel_blend = simulation_params->dynamic_params.alignment_coeff * utils::math::normalize_or_zero(alignment)
				+ simulation_params->dynamic_params.cohesion_coeff * utils::math::normalize_or_zero(cohesion)
				+ simulation_params->dynamic_params.separation_coeff * utils::math::normalize_or_zero(separation)
				+ simulation_params->dynamic_params.wall_separation_coeff * utils::math::normalize_or_zero(wall_separation);

			ssbo_velocities[current] = utils::math::normalize_or_zero(ssbo_velocities[current] + accel_blend * delta_time); //v = u + at
			ssbo_positions [current] += ssbo_velocities[current] * simulation_params->dynamic_params.boid_speed * delta_time; //s = vt
			ssbo_positions [current] = clamp(ssbo_positions[current], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
		}
	}

	namespace grid
	{
		namespace uniform
		{
			inline __global__ void alignment(float4* alignments, float4* positions, float4* velocities, size_t amount, boid_cell_index* boid_cell_indices, idx_range* cell_idx_ranges, size_t max_radius)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= amount) return; // avoid threads ids overflowing the array
				
				float4 alignment{ 0 };
				int current_neighbour;
				boid_cell_index current_boid = boid_cell_indices[idx];
				idx_range range = cell_idx_ranges[current_boid.cell_id];
				bool in_radius;
				for (size_t i = range.start; i < range.end; i++)
				{
					current_neighbour = boid_cell_indices[i].boid_id;
					in_radius = utils::math::distance2(positions[current_boid.boid_id], positions[current_neighbour]) < max_radius * max_radius;
					// condition as multiplier avoids warp divergence
					alignment += velocities[current_neighbour] * in_radius;
				}

				alignments[current_boid.boid_id] = utils::math::normalize_or_zero(alignment);

			}

			inline __global__ void cohesion(float4* cohesions, float4* positions, size_t amount, boid_cell_index* boid_cell_indices, idx_range* cell_idx_ranges, size_t max_radius)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= amount) return; 

				float4 cohesion{ 0 }, baricenter{ 0 };
				float counter{ 0 };
				int current_neighbour;
				boid_cell_index current_boid = boid_cell_indices[idx];
				idx_range range = cell_idx_ranges[current_boid.cell_id];
				bool in_radius;
				for (size_t i = range.start; i < range.end; i++)
				{
					current_neighbour = boid_cell_indices[i].boid_id;
					in_radius = utils::math::distance2(positions[current_boid.boid_id], positions[current_neighbour]) < max_radius * max_radius;

					baricenter += positions[current_neighbour] * in_radius;
					counter += 1.f * in_radius;
				}
				baricenter /= counter;
				cohesion = baricenter - positions[current_boid.boid_id];
				cohesions[current_boid.boid_id] = utils::math::normalize_or_zero(cohesion);

			}

			inline __global__ void separation(float4* separations, float4* positions, size_t amount, boid_cell_index* boid_cell_indices, idx_range* cell_idx_ranges, size_t max_radius)
			{
				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= amount) return;

				float4 separation{ 0 };
				float4 repulsion; float repulsion_length2;
				bool in_radius;

				int current_neighbour;
				boid_cell_index current_boid = boid_cell_indices[idx];
				idx_range range = cell_idx_ranges[current_boid.cell_id];
				for (size_t i = range.start; i < range.end; i++)
				{
					current_neighbour = boid_cell_indices[i].boid_id;
					repulsion = positions[current_boid.boid_id] - positions[current_neighbour];
					repulsion_length2 = utils::math::length2(repulsion);
					in_radius = repulsion_length2 < max_radius * max_radius;
					separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;
				}

				separations[current_boid.boid_id] = utils::math::normalize_or_zero(separation);
			}
		
			inline __global__ void flock(float4* ssbo_positions, float4* ssbo_velocities, size_t amount,
				boid_cell_index* boid_cell_indices, idx_range* cell_idx_ranges, size_t max_radius,
				utils::math::plane* borders,
				utils::runners::boid_runner::simulation_parameters* simulation_params, const float delta_time)
			{
				size_t current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= amount) return;

				float4 alignment{ 0 }, cohesion{ 0 }, separation{ 0 }, wall_separation{ 0 };
				float4 baricenter{ 0 };
				float counter{ 0 };
				float4 repulsion{ 0 }; float repulsion_length2{ 0 };
				float4 wall_repulsion{ 0 }; float wall_distance{ 0 };

				int current_neighbour;
				boid_cell_index current_boid = boid_cell_indices[current];
				idx_range range = cell_idx_ranges[current_boid.cell_id];
				bool in_radius, near_wall;
				float chs = simulation_params->static_params.cube_size / 2; //chs -> cube half size
				float wall_repel_distance = 1 + (chs * 2) * .01f;

				for (size_t i = range.start; i < range.end; i++)
				{
					current_neighbour = boid_cell_indices[i].boid_id;
					in_radius = utils::math::distance2(ssbo_positions[current_boid.boid_id], ssbo_positions[current_neighbour]) < max_radius * max_radius;

					// alignment
					alignment += ssbo_velocities[current_neighbour] * in_radius;

					// cohesion
					baricenter += ssbo_positions[current_neighbour] * in_radius;
					counter += 1.f * in_radius;

					// separation
					repulsion = ssbo_positions[current_boid.boid_id] - ssbo_positions[current_neighbour];
					repulsion_length2 = utils::math::length2(repulsion);
					separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;

					// wall_separation
					for (size_t b = 0; b < 6; b++)
					{
						wall_distance = utils::math::distance_point_plane(ssbo_positions[current_boid.boid_id], borders[b]) + 0.0001f;
						near_wall = wall_distance < wall_repel_distance;
						wall_repulsion = (borders[b].normal / abs(wall_distance)) * near_wall;
						wall_separation += wall_repulsion;
					}
				}
				baricenter /= counter;
				cohesion = baricenter - ssbo_positions[current_boid.boid_id];

				float4 accel_blend = simulation_params->dynamic_params.alignment_coeff * utils::math::normalize_or_zero(alignment)
					+ simulation_params->dynamic_params.cohesion_coeff * utils::math::normalize_or_zero(cohesion)
					+ simulation_params->dynamic_params.separation_coeff * utils::math::normalize_or_zero(separation)
					+ simulation_params->dynamic_params.wall_separation_coeff * utils::math::normalize_or_zero(wall_separation);

				ssbo_velocities[current_boid.boid_id] = utils::math::normalize_or_zero(ssbo_velocities[current_boid.boid_id] + accel_blend * delta_time); //v = u + at
				ssbo_positions [current_boid.boid_id] += ssbo_velocities[current_boid.boid_id] * simulation_params->dynamic_params.boid_speed * delta_time; //s = vt
				ssbo_positions [current_boid.boid_id] = clamp(ssbo_positions[current_boid.boid_id], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
			}
		}

		namespace coherent
		{
			inline __global__ void alignment(float4* alignments, float4* positions, float4* velocities, int* cell_ids, size_t amount, idx_range* cell_idx_ranges, size_t max_radius)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= amount) return; // avoid threads ids overflowing the array
			
				float4 alignment{ 0 };
				idx_range range = cell_idx_ranges[cell_ids[current]];
				bool in_radius;

				//size_t range_diff = range.end - range.start;
				//float4* cell_positions = (float4*)malloc(sizeof(float4) * 1);
				//memcpy(cell_positions, positions, sizeof(float4) * 1);
				//free(cell_positions);

				for (size_t i = range.start; i < range.end; i++)
				{
					in_radius = utils::math::distance2(positions[current], positions[i]) < max_radius * max_radius;
					// condition as multiplier avoids warp divergence
					alignment += velocities[i] * in_radius;
				}
			
				alignments[current] = utils::math::normalize_or_zero(alignment);
				
			}

			inline __global__ void cohesion(float4* cohesions, float4* positions, int* cell_ids, size_t amount, idx_range* cell_idx_ranges, size_t max_radius)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= amount) return;

				float4 cohesion{ 0 }, baricenter{ 0 };
				float counter{ 0 };
				idx_range range = cell_idx_ranges[cell_ids[current]];
				bool in_radius;

				for (size_t i = range.start; i < range.end; i++)
				{
					in_radius = utils::math::distance2(positions[current], positions[i]) < max_radius * max_radius;

					baricenter += positions[i] * in_radius;
					counter += 1.f * in_radius;
				}
				baricenter /= counter;
				cohesion = baricenter - positions[current];

				cohesions[current] = utils::math::normalize_or_zero(cohesion);
			}

			inline __global__ void separation(float4* separations, float4* positions, int* cell_ids, size_t amount, idx_range* cell_idx_ranges, size_t max_radius)
			{
				int current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= amount) return;

				float4 separation{ 0 };
				float4 repulsion; float repulsion_length2;
				bool in_radius;

				idx_range range = cell_idx_ranges[cell_ids[current]];
				for (size_t i = range.start; i < range.end; i++)
				{
					repulsion = positions[current] - positions[i];
					repulsion_length2 = utils::math::length2(repulsion); 
					in_radius = repulsion_length2 < max_radius * max_radius;
					separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;
					//separation += (utils::math::normalize_or_zero(repulsion) / (length(repulsion) + 0.0001f)) * in_radius; //old formula
				}

				separations[current] = utils::math::normalize_or_zero(separation);
			}

			// Single master behaviour including alignment, cohesion, separation and wall separation in one go: 
			// not modular but more efficient than separate behaviours because requires less global memory loads
			inline __global__ void flock(float4* ssbo_positions, float4* ssbo_velocities, 
				float4* sorted_positions, float4* sorted_velocities, int* cell_ids, size_t amount,
				int* cell_idx_range_start_dptr, int* cell_idx_range_end_dptr, size_t max_radius,
				//utils::math::plane* borders, // TODO THIS CAN GO IN constant memory SMEM
				utils::runners::boid_runner::simulation_parameters* simulation_params, // TODO THIS CAN GO IN SMEM
				const float delta_time)
			{
				size_t current = blockIdx.x * blockDim.x + threadIdx.x;
				if (current >= amount) return;

				float4 alignment{ 0 }, cohesion{ 0 }, separation{ 0 }, wall_separation{ 0 };
				//float4 baricenter{ 0 };
				float counter{ 0 };
				float4 repulsion{ 0 }; float repulsion_length2{ 0 };
				float wall_distance{ 0 };

				int boid_cell = cell_ids[current];
				int range_start = cell_idx_range_start_dptr[boid_cell], range_end = cell_idx_range_end_dptr[boid_cell];
				bool in_radius, near_wall;
				utils::runners::boid_runner::simulation_parameters sim_params = *simulation_params; 
				float chs = sim_params.static_params.cube_size / 2; //chs -> cube half size
				float wall_repel_distance = 1 + (chs * 2) * .01f;

				float4 position_current = sorted_positions[current], position_i;
				float4 new_position, new_velocity;

				for (size_t i = range_start; i < range_end; i++)
				{
					position_i = sorted_positions[i];
					in_radius = utils::math::distance2(position_current, position_i) < max_radius * max_radius;

					// alignment
					alignment += sorted_velocities[i] * in_radius;

					// cohesion
					cohesion += position_i * in_radius;
					counter += 1.f * in_radius;

					// separation
					repulsion = position_current - position_i;
					repulsion_length2 = utils::math::length2(repulsion);
					separation += (repulsion / (repulsion_length2 + 0.0001f)) * in_radius;

					// wall_separation
					for (size_t b = 0; b < 6; b++)
					{
						wall_distance = utils::math::distance_point_plane(position_current, sim_volume_cdptr[b]) + 0.0001f;
						near_wall = wall_distance < wall_repel_distance;
						wall_separation += (sim_volume_cdptr[b].normal / abs(wall_distance)) * near_wall; //wall repulsion calculation
					}
				}
				cohesion /= counter;
				cohesion = cohesion - position_current;

				float4 accel_blend = sim_params.dynamic_params.alignment_coeff * utils::math::normalize_or_zero(alignment)
					+ sim_params.dynamic_params.cohesion_coeff * utils::math::normalize_or_zero(cohesion)
					+ sim_params.dynamic_params.separation_coeff * utils::math::normalize_or_zero(separation)
					+ sim_params.dynamic_params.wall_separation_coeff * utils::math::normalize_or_zero(wall_separation);

				new_velocity = utils::math::normalize_or_zero(sorted_velocities[current] + accel_blend * delta_time); //v = u + at
				new_position = position_current + new_velocity * sim_params.dynamic_params.boid_speed * delta_time; //s = vt
				new_position = clamp(new_position, { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube

				ssbo_velocities[current] = new_velocity;
				ssbo_positions [current] = new_position;
				// TODO check with nvprof if better idk
				//ssbo_velocities[current] = utils::math::normalize_or_zero(ssbo_velocities[current] + accel_blend * delta_time); //v = u + at
				//ssbo_positions [current] += ssbo_velocities[current] * simulation_params->dynamic_params.boid_speed * delta_time; //s = vt
				//ssbo_positions [current] = clamp(ssbo_positions[current], { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
			}

			inline __global__ void blender(float4* ssbo_positions, float4* ssbo_velocities,
				float4* sorted_positions, float4* sorted_velocities,
				float4* alignments, float4* cohesions, float4* separations, float4* wall_separations,
				utils::runners::boid_runner::simulation_parameters* simulation_params, size_t amount, const float delta_time)
			{
				int i = blockIdx.x * blockDim.x + threadIdx.x;
				if (i >= amount) return;

				float chs = simulation_params->static_params.cube_size / 2;
				float4 accel_blend, new_position = sorted_positions[i], new_velocity;

				accel_blend = simulation_params->dynamic_params.alignment_coeff * alignments[i]
					+ simulation_params->dynamic_params.cohesion_coeff * cohesions[i]
					+ simulation_params->dynamic_params.separation_coeff * separations[i]
					+ simulation_params->dynamic_params.wall_separation_coeff * wall_separations[i];

				new_velocity = utils::math::normalize_or_zero(sorted_velocities[i] + accel_blend * delta_time); //v = u + at
				new_position += new_velocity * simulation_params->dynamic_params.boid_speed * delta_time; //s = vt
				new_position = clamp(new_position, { -chs,-chs,-chs,0 }, { chs,chs,chs,0 }); // ensures boids remain into the cube
			
				ssbo_velocities[i] = new_velocity;
				ssbo_positions [i] = new_position;
			}
		}
	}
}
