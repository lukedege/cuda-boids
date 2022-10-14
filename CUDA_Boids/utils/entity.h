#pragma once
#pragma once

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "mesh.h"
#include "shader.h"

namespace utils::graphics::opengl
{

	// Object in scene
	class Entity
	{
		Mesh mesh;
		size_t  amount      { 1 };
		glm::mat4 transform { 1 };
		glm::mat3 normal    { 1 };
		bool      cumulate  { false };

	public:
		Entity(std::vector<Vertex>& v, std::vector<GLuint>& i, size_t amount = 1, const glm::mat4& transform = glm::mat4{ 1 }) :
			mesh{ v,i }, amount{ amount }, transform{ transform }, normal{ 1 }, cumulate{ false } {}
		
		Entity(Mesh& other, size_t amount = 1, const glm::mat4& transform = glm::mat4{ 1 }) :
			mesh{ std::move(other) }, amount{ amount }, transform{ transform }, normal{ 1 }, cumulate{ false } {}

		// Local space transformations!!
		void scale     (glm::vec3 scaling)                         { transform = glm::scale    (transform, scaling);                 }
		void translate (glm::vec3 translation)                     { transform = glm::translate(transform, translation);             }
		void rotate    (float angle_rad, glm::vec3 rotationAxis)   { transform = glm::rotate   (transform, angle_rad, rotationAxis); }
		void rotate_deg(float angle_deg, glm::vec3 rotationAxis)   { rotate(glm::radians(angle_deg), rotationAxis);                  }

		void draw(const Shader& shader, const glm::mat4& viewProjection)
		{
			shader.use();
			recomputeNormal(viewProjection);

			shader.setMat4("model_matrix", transform);
			shader.setMat3("normal_matrix", normal);

			mesh.draw_instanced(amount);

			if (!cumulate)
			{
				// reset to identity
				transform = glm::mat4(1);
				normal = glm::mat3(1);
			}
		}

		void toggle_cumulation()
		{
			cumulate = !cumulate;
		}

	private:
		void recomputeNormal(glm::mat4 viewProjection) { normal = glm::inverseTranspose(glm::mat3(viewProjection * transform)); }
	};
}