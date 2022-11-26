#pragma once

#include <vector>
#include <string>
#include <iostream>

#include <glad.h>
#include <glm/glm.hpp>

namespace utils::graphics::opengl
{
	struct Vertex
	{
		glm::vec3 position;
		//other properties
	};

	class Mesh
	{
	public:
		std::vector<Vertex> vertices;
		std::vector<GLuint> indices;
		GLuint VAO;

		Mesh(std::vector<Vertex>& v, std::vector<GLuint>& i) noexcept :
			vertices(std::move(v)), indices(std::move(i)) {
			setupMesh();
		}

		Mesh(const Mesh& copy) = delete;
		Mesh& operator=(const Mesh& copy) = delete;

		Mesh(Mesh&& move) noexcept :
			vertices(std::move(move.vertices)), indices(std::move(move.indices)),
			VAO(move.VAO), VBO(move.VBO), EBO(move.EBO)
		{
			move.VAO = 0;
		}

		Mesh& operator=(Mesh&& move) noexcept
		{
			freeGPU();

			// Check if it exists
			if (move.VAO)
			{
				vertices = std::move(move.vertices);
				indices = std::move(move.indices);
				VAO = move.VAO; VBO = move.VBO; EBO = move.EBO;

				move.VAO = 0;
			}
			else
			{
				VAO = 0;
			}

			return *this;
		}

		~Mesh()
		{
			freeGPU();
		}

		void draw(GLenum mode = GL_TRIANGLES) const
		{
			glBindVertexArray(VAO);
			glDrawElements(mode, indices.size(), GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
		}

		void draw_instanced(int amount, GLenum mode = GL_TRIANGLES)
		{
			glBindVertexArray(VAO);
			glDrawElementsInstanced(mode, indices.size(), GL_UNSIGNED_INT, 0, amount);
			glBindVertexArray(0);
		}

	private:
		GLuint VBO, EBO;

		void setupMesh()
		{
			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			glGenBuffers(1, &EBO);

			// VAO is made "active"    
			glBindVertexArray(VAO);
			// we copy data in the VBO - we must set the data dimension, and the pointer to the structure cointaining the data
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
			// we copy data in the EBO - we must set the data dimension, and the pointer to the structure cointaining the data
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

			// positions (location = 0 in shader)
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)0);

			glBindBuffer(GL_ARRAY_BUFFER, 0); // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound vertex buffer object so afterwards we can safely unbind
			glBindVertexArray(0); // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO

		}

		void freeGPU()
		{
			// Check if we have something in GPU
			if (VAO)
			{
				glDeleteVertexArrays(1, &VAO);
				glDeleteBuffers(1, &VBO);
				glDeleteBuffers(1, &EBO);
			}
		}
	};
}