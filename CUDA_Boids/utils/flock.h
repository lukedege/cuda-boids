#pragma once

#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mesh.h"
#include "shader.h"

namespace utils::graphics::opengl
{
	// Instanced drawing of the same entity with different positions
	class Flock : public Entity
	{
	public:
		std::vector<glm::vec2> positions;

		Flock(std::vector<Vertex>& v, std::vector<GLuint>& i, size_t amount = 1, const glm::mat4& transform = glm::mat4{ 1 }) :
			Entity{ v, i, transform }, positions{} 
		{
			positions.resize(amount);
		}
		
		Flock(Mesh& other, size_t amount = 1, const glm::mat4& transform = glm::mat4{ 1 }) :
			Entity{ other, transform }, positions{}
		{
			positions.resize(amount);
		}

		void draw(const Shader& shader, const glm::mat4& viewProjection)
		{
			shader.use();
			recompute_normal(viewProjection);

			shader.setMat4("model_matrix", transform);
			shader.setMat3("normal_matrix", normal);

			//for (size_t i = 0; i < positions.size(); i++)
			//{
			//	shader.setVec2("positions[" + std::to_string(i) + "]", { positions[i].x, positions[i].y });
			//}

			mesh.draw_instanced(positions.size());

			if (!cumulate)
			{
				// reset to identity
				transform = glm::mat4(1);
				normal = glm::mat3(1);
			}
		}
	};
}