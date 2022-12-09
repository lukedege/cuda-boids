#pragma once

#include "../utils/utils.h"
#include "../utils/CUDA/vector_math.h"

namespace utils::runners::behaviours::grid
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
}