#pragma once
#include <glm/glm.hpp>

#include "../utils/shader.h"

namespace utils::runners
{
	class boid_runner
	{
	public:
		virtual void calculate(const float delta_time) = 0;
		virtual void draw(const glm::mat4& view_matrix, const glm::mat4& projection_matrix) = 0;
	};
}