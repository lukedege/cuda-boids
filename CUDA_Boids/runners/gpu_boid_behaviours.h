#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../utils/utils.h"
#include "../utils/CUDA/vector_math.h"
#include "boid_runner.h"

namespace utils::runners::behaviours::gpu
{
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

		inline __global__ void wall_separation(float4* wall_separations, float4* positions, utils::math::plane* borders, size_t amount)
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
	}
	namespace uniform_grid
	{
		// TODO
	}
	namespace coherent_grid
	{
		// TODO
	}
}
