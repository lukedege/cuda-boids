#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "../utils/CUDA/vector_math.h"

namespace utils::runners::behaviours::cpu::naive
{
	//TODO check if behaviours are CORRECT
	inline glm::vec4 alignment(size_t current, glm::vec4* positions, glm::vec4* velocities, size_t amount, size_t max_radius)
	{
		glm::vec4 alignment{ 0 };
		bool in_radius;
		for (size_t i = 0; i < amount; i++)
		{
			in_radius = glm::distance2(positions[current], positions[i]) < max_radius * max_radius;
			if(in_radius)
				alignment += velocities[i];
		}

		return utils::math::normalize(alignment);
	}

	inline glm::vec4 cohesion(size_t current, glm::vec4* positions, size_t amount, size_t max_radius)
	{
		glm::vec4 cohesion{ 0 }, baricenter{ 0 };
		float counter{ 0 };
		bool in_radius;
		for (size_t i = 0; i < amount; i++)
		{
			in_radius = glm::distance2(positions[current], positions[i]) < max_radius * max_radius;
			if (in_radius)
			{
				baricenter += positions[i];
				counter += 1.f;
			}
		}
		baricenter /= counter;
		cohesion = baricenter - positions[current];
		return utils::math::normalize(cohesion);
	}

	inline glm::vec4 separation(size_t current, glm::vec4* positions, size_t amount, size_t max_radius)
	{
		glm::vec4 separation{ 0 };
		glm::vec4 repulsion;
		bool in_radius;
		// boid check
		for (size_t i = 0; i < amount; i++)
		{
			in_radius = glm::distance2(positions[current], positions[i]) < max_radius * max_radius;
			if (in_radius)
			{
				repulsion = positions[current] - positions[i]; // TODO sottrazione e distance2 sono ridondanti, ottimizza
				separation += utils::math::normalize(repulsion) / (glm::length(repulsion) + 0.0001f);
			}
		}

		return utils::math::normalize(separation);
	}

	inline glm::vec4 wall_separation(size_t current, glm::vec4* positions, utils::math::plane* borders, size_t amount)
	{
		glm::vec4 separation{ 0 };
		glm::vec4 repulsion;
		float distance;
		// wall check
		for (size_t i = 0; i < amount; i++)
		{
			for (size_t b = 0; b < 6; b++)
			{
				distance = utils::math::distance_point_plane(positions[current], borders[b]) + 0.0001f;
				//if distance < 1 
				//distance = distance * 0.01f;
				repulsion = borders[b].normal / abs(distance);
				separation += repulsion;
			}
		}

		return utils::math::normalize(separation);
	}
}

