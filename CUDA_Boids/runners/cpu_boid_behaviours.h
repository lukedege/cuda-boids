#pragma once

#include "../utils/utils.h"
#include "../utils/CUDA/vector_math.h"

namespace utils::runners::behaviours::cpu
{
	namespace naive
	{
		inline float4 alignment(size_t current, float4* positions, float4* velocities, size_t amount, size_t max_radius)
		{
			float4 alignment{ 0 };
			bool in_radius;
			for (size_t i = 0; i < amount; i++)
			{
				in_radius = utils::math::distance2(positions[current], positions[i]) < max_radius * max_radius;
				if (in_radius)
					alignment += velocities[i];
			}

			return utils::math::normalize_or_zero(alignment);
		}

		inline float4 cohesion(size_t current, float4* positions, size_t amount, size_t max_radius)
		{
			float4 cohesion{ 0 }, baricenter{ 0 };
			float counter{ 0 };
			bool in_radius;
			for (size_t i = 0; i < amount; i++)
			{
				in_radius = utils::math::distance2(positions[current], positions[i]) < max_radius * max_radius;
				if (in_radius)
				{
					baricenter += positions[i];
					counter += 1.f;
				}
			}
			baricenter /= counter;
			cohesion = baricenter - positions[current];
			return utils::math::normalize_or_zero(cohesion);
		}

		inline float4 separation(size_t current, float4* positions, size_t amount, size_t max_radius)
		{
			float4 separation{ 0 };
			float4 repulsion;
			float in_radius;
			// boid check
			for (size_t i = 0; i < amount; i++)
			{
				repulsion = positions[current] - positions[i];
				in_radius = utils::math::length2(repulsion) < max_radius * max_radius;
				if (in_radius)
					separation += (utils::math::normalize_or_zero(repulsion) / (length(repulsion) + 0.0001f));
			}

			return utils::math::normalize_or_zero(separation);
		}

		inline float4 wall_separation(size_t current, float4* positions, utils::math::plane* borders, size_t amount)
		{
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


			return utils::math::normalize_or_zero(separation);
		}
	}
	namespace uniform_grid
	{
		struct boid_cell_index
		{
			int boid_id;
			int cell_id;
		};

		inline float4 alignment(size_t current, float4* positions, float4* velocities, boid_cell_index* boid_cell_indices, int start, int end, size_t max_radius)
		{
			float4 alignment{ 0 };
			int current_neighbour;
			float in_radius;
			for (size_t i = start; i < end; i++)
			{
				current_neighbour = boid_cell_indices[i].boid_id;
				in_radius = utils::math::distance2(positions[current], positions[current_neighbour]) < max_radius * max_radius;
				if (in_radius)
					alignment += velocities[current_neighbour];
			}
			return utils::math::normalize_or_zero(alignment);
		}

		inline float4 cohesion(size_t current, float4* positions, boid_cell_index* boid_cell_indices, int start, int end, size_t max_radius)
		{
			float4 cohesion{ 0 }, baricenter{ 0 };
			float counter{ 0 };
			int current_neighbour;
			float in_radius;
			for (size_t i = start; i < end; i++)
			{
				current_neighbour = boid_cell_indices[i].boid_id;
				in_radius = utils::math::distance2(positions[current], positions[current_neighbour]) < max_radius * max_radius;
				if (in_radius)
				{
					baricenter += positions[current_neighbour];
					counter += 1.f;
				}
			}
			baricenter /= counter;
			cohesion = baricenter - positions[current];
			return utils::math::normalize_or_zero(cohesion);
		}

		inline float4 separation(size_t current, float4* positions, boid_cell_index* boid_cell_indices, int start, int end, size_t max_radius)
		{
			float4 separation{ 0 };
			float4 repulsion;
			int current_neighbour;
			float in_radius;
			for (size_t i = start; i < end; i++)
			{
				current_neighbour = boid_cell_indices[i].boid_id;
				repulsion = positions[current] - positions[current_neighbour];
				in_radius = utils::math::length2(repulsion) < max_radius * max_radius;
				if (in_radius)
					separation += (utils::math::normalize_or_zero(repulsion) / (length(repulsion) + 0.0001f));
			}
			return utils::math::normalize_or_zero(separation);
		}

		inline float4 wall_separation(size_t current, float4* positions, utils::math::plane* borders)
		{
			float4 separation{ 0 };
			float4 repulsion;
			float distance;
			float near_wall;
			for (size_t b = 0; b < 6; b++)
			{
				distance = utils::math::distance_point_plane(positions[current], borders[b]) + 0.0001f;
				near_wall = distance < 1.f;
				repulsion = (borders[b].normal / abs(distance)) * near_wall;
				separation += repulsion;
			}

			return utils::math::normalize_or_zero(separation);
		}
	}
	namespace coherent_grid
	{
		struct boid_cell_index
		{
			int boid_id;
			int cell_id;
		};

		struct idx_range
		{
			int start{ 0 }; //inclusive
			int end{ 0 }; //exclusive
		};

		inline float4 alignment(size_t current, float4* positions, float4* velocities, int* cell_ids, idx_range* cell_idx_ranges, size_t max_radius)
		{
			float4 alignment{ 0 };
			float in_radius;
			idx_range idx_range = cell_idx_ranges[cell_ids[current]];
			size_t start = idx_range.start;
			size_t end = idx_range.end;
			for (size_t i = start; i < end; i++)
			{
				in_radius = utils::math::distance2(positions[current], positions[i]) < max_radius * max_radius;
				if (in_radius)
					alignment += velocities[i];
			}
			return utils::math::normalize_or_zero(alignment);
		}

		inline float4 cohesion(size_t current, float4* positions, int* cell_ids, idx_range* cell_idx_ranges, size_t max_radius)
		{
			float4 cohesion{ 0 }, baricenter{ 0 };
			float counter{ 0 };
			float in_radius;
			idx_range idx_range = cell_idx_ranges[cell_ids[current]];
			size_t start = idx_range.start;
			size_t end = idx_range.end;
			for (size_t i = start; i < end; i++)
			{
				in_radius = utils::math::distance2(positions[current], positions[i]) < max_radius * max_radius;
				if (in_radius)
				{
					baricenter += positions[i];
					counter += 1.f;
				}
			}
			baricenter /= counter;
			cohesion = baricenter - positions[current];
			return utils::math::normalize_or_zero(cohesion);
		}

		inline float4 separation(size_t current, float4* positions, int* cell_ids, idx_range* cell_idx_ranges, size_t max_radius)
		{
			float4 separation{ 0 };
			float4 repulsion;
			float in_radius;
			idx_range idx_range = cell_idx_ranges[cell_ids[current]];
			size_t start = idx_range.start;
			size_t end = idx_range.end;
			for (size_t i = start; i < end; i++)
			{
				repulsion = positions[current] - positions[i];
				in_radius = utils::math::length2(repulsion) < max_radius * max_radius;
				if (in_radius)
					separation += (utils::math::normalize_or_zero(repulsion) / (length(repulsion) + 0.0001f));
			}
			return utils::math::normalize_or_zero(separation);
		}

		inline float4 wall_separation(size_t current, float4* positions, utils::math::plane* borders)
		{
			float4 separation{ 0 };
			float4 repulsion;
			float distance;
			float near_wall;
			for (size_t b = 0; b < 6; b++)
			{
				distance = utils::math::distance_point_plane(positions[current], borders[b]) + 0.0001f;
				near_wall = distance < 1.f;
				repulsion = (borders[b].normal / abs(distance)) * near_wall;
				separation += repulsion;
			}

			return utils::math::normalize_or_zero(separation);
		}
	}
}
