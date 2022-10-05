#pragma once
/*
   Shader class
   - loading Shader source code, Shader Program creation
*/

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#include <glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace utils::graphics::opengl
{

	class Shader
	{
	public:
		GLuint program;

		Shader(const GLchar* vertPath, const GLchar* fragPath, std::vector<const GLchar*> utilPaths = {}, GLuint glMajor = 4, GLuint glMinor = 1) :
			program{ glCreateProgram() }, glMajorVersion{ glMajor }, glMinorVersion{ glMinor }
		{
			const std::string vertSource = loadSource(vertPath);
			const std::string fragSource = loadSource(fragPath);

			std::string utilsSource;

			for (const GLchar* utilPath : utilPaths)
			{
				utilsSource += loadSource(utilPath) + "\n";
			}

			GLuint   vertexShader = compileShader(vertSource, GL_VERTEX_SHADER, utilsSource);
			GLuint fragmentShader = compileShader(fragSource, GL_FRAGMENT_SHADER, utilsSource);

			glAttachShader(program, vertexShader);
			glAttachShader(program, fragmentShader);

			try {
				glLinkProgram(program);
			}
			catch (std::exception e) { auto x = glGetError(); checkLinkingErrors(); std::cout << e.what(); }

			glDeleteShader(vertexShader);
			glDeleteShader(fragmentShader);
		}

		~Shader() { glDeleteProgram(program); }

		void use() const noexcept { glUseProgram(program); }

		std::vector<std::string> findSubroutines(GLenum shaderType)
		{
			std::vector<std::string> ret;

			int maxSub = 0, maxSubU = 0, countActiveSU = 0;
			GLchar name[256];
			int len = 0, numCompS = 0;

			// global parameters about the Subroutines parameters of the system
			glGetIntegerv(GL_MAX_SUBROUTINES, &maxSub);
			glGetIntegerv(GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS, &maxSubU);
			std::cout << "Max Subroutines:" << maxSub << " - Max Subroutine Uniforms:" << maxSubU << std::endl;

			// get the number of Subroutine uniforms for the kind of shader used
			glGetProgramStageiv(program, shaderType, GL_ACTIVE_SUBROUTINE_UNIFORMS, &countActiveSU);

			// print info for every Subroutine uniform
			for (int i = 0; i < countActiveSU; i++) {

				// get the name of the Subroutine uniform (in this example, we have only one)
				glGetActiveSubroutineUniformName(program, shaderType, i, 256, &len, name);
				// print index and name of the Subroutine uniform
				std::cout << "Subroutine Uniform: " << i << " - name: " << name << std::endl;

				// get the number of subroutines
				glGetActiveSubroutineUniformiv(program, shaderType, i, GL_NUM_COMPATIBLE_SUBROUTINES, &numCompS);

				// get the indices of the active subroutines info and write into the array s
				int* s = new int[numCompS];
				glGetActiveSubroutineUniformiv(program, shaderType, i, GL_COMPATIBLE_SUBROUTINES, s);
				std::cout << "Compatible Subroutines:" << std::endl;

				// for each index, get the name of the subroutines, print info, and save the name in the shaders vector
				for (int j = 0; j < numCompS; ++j) {
					glGetActiveSubroutineName(program, shaderType, s[j], 256, &len, name);
					std::cout << "\t" << s[j] << " - " << name << "\n";
					ret.push_back(name);
				}
				std::cout << std::endl;

				delete[] s;
			}

			return ret;
		}

		GLuint getSubroutineIndex(GLenum shaderType, const char* subroutineName)
		{
			return glGetSubroutineIndex(program, shaderType, subroutineName);
		}

#pragma region utility_uniform_functions
		void setBool (const std::string& name, bool value)                            const { glUniform1i(glGetUniformLocation(program, name.c_str()), (int)value); }
		void setInt  (const std::string& name, int value)                             const { glUniform1i(glGetUniformLocation(program, name.c_str()), value); }
		void setUint (const std::string& name, unsigned int value)                    const { glUniform1ui(glGetUniformLocation(program, name.c_str()), value); }
		void setFloat(const std::string& name, float value)                           const { glUniform1f(glGetUniformLocation(program, name.c_str()), value); }

		void setVec2 (const std::string& name, const GLfloat value[])                 const { glUniform2fv(glGetUniformLocation(program, name.c_str()), 1, &value[0]); }
		void setVec2 (const std::string& name, const glm::vec2& value)                const { glUniform2fv(glGetUniformLocation(program, name.c_str()), 1, glm::value_ptr(value)); }
		void setVec2 (const std::string& name, float x, float y)                      const { glUniform2f(glGetUniformLocation(program, name.c_str()), x, y); }
					 
		void setVec3 (const std::string& name, const GLfloat value[])                 const { glUniform3fv(glGetUniformLocation(program, name.c_str()), 1, &value[0]); }
		void setVec3 (const std::string& name, const glm::vec3& value)                const { glUniform3fv(glGetUniformLocation(program, name.c_str()), 1, glm::value_ptr(value)); }
		void setVec3 (const std::string& name, float x, float y, float z)             const { glUniform3f(glGetUniformLocation(program, name.c_str()), x, y, z); }
					 
		void setVec4 (const std::string& name, const GLfloat value[])                 const { glUniform4fv(glGetUniformLocation(program, name.c_str()), 1, &value[0]); }
		void setVec4 (const std::string& name, const glm::vec4& value)                const { glUniform4fv(glGetUniformLocation(program, name.c_str()), 1, glm::value_ptr(value)); }
		void setVec4 (const std::string& name, float x, float y, float z, float w)    const { glUniform4f(glGetUniformLocation(program, name.c_str()), x, y, z, w); }
					 
		void setMat2 (const std::string& name, const glm::mat2& mat)                  const { glUniformMatrix2fv(glGetUniformLocation(program, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat)); }

		void setMat3 (const std::string& name, const glm::mat3& mat)                  const { glUniformMatrix3fv(glGetUniformLocation(program, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat)); }
					 
		void setMat4 (const std::string& name, const glm::mat4& mat)                  const { glUniformMatrix4fv(glGetUniformLocation(program, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat)); }
#pragma endregion 

	private:
		GLuint glMajorVersion;
		GLuint glMinorVersion;

		const std::string loadSource(const GLchar* sourcePath) const noexcept
		{
			std::string         sourceCode;
			std::ifstream       sourceFile;
			std::stringstream sourceStream;

			sourceFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

			try
			{
				sourceFile.open(sourcePath);
				sourceStream << sourceFile.rdbuf();
				sourceFile.close();
				sourceCode = sourceStream.str();
			}
			catch (const std::exception& e)
			{
				std::cerr << e.what() << '\n';
			}

			return sourceCode;
		}

		GLuint compileShader(const std::string& shaderSource, GLenum shaderType, const std::string& utilsSource = "") const noexcept
		{
			std::string mergedSource = "";
			// Prepending version
			mergedSource += "#version " + std::to_string(glMajorVersion) + std::to_string(glMinorVersion) + "0 core\n";

			// Prepending utils
			mergedSource += utilsSource + shaderSource;

			const std::string finalSource{ mergedSource };

			// Shader creation
			GLuint shader = glCreateShader(shaderType);
			const GLchar* c = finalSource.c_str(); // as of documentation, c_str return a null terminated string
			
			glShaderSource(shader, 1, &c, NULL);   // as of documentation, glShaderSource is supposed to copy the content of the c string so we can trash it afterwards
			glCompileShader(shader);
			checkCompileErrors(shader);
			return shader;
		}

		void checkCompileErrors(GLuint shader) const noexcept
		{
			// Check for compile time errors TODO
			GLint success; GLchar infoLog[512];
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
			if (!success)
			{
				glGetShaderInfoLog(shader, 512, NULL, infoLog);
				std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
			}
		}

		void checkLinkingErrors() const noexcept
		{
			GLint success; GLchar infoLog[512];
			glGetProgramiv(program, GL_LINK_STATUS, &success);
			if (!success) {
				glGetProgramInfoLog(program, 512, NULL, infoLog);
				std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			}
		}
	};
}