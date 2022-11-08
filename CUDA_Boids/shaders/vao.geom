// Geometry shader
// #version 430 core
layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 velocity_norm[];

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

void main()
{
	mat4 vp = projection_matrix * view_matrix;

	//arrowhead
	gl_Position = vp * (gl_in[0].gl_Position + vec4(velocity_norm[0], 0.0f));// * view_matrix * projection_matrix;
	EmitVertex();

	//calculate orthogonal for triangle base
	vec3 orthogonal_vel = cross(velocity_norm[0], vec3(0,0,1)) * 0.4f;

	//base-left
	gl_Position = vp * (gl_in[0].gl_Position + vec4(orthogonal_vel, 0.0f));// * view_matrix * projection_matrix;
	EmitVertex();
	
	//base-right
	gl_Position = vp * (gl_in[0].gl_Position - vec4(orthogonal_vel, 0.0f));// * view_matrix * projection_matrix;
	EmitVertex();
	
	EndPrimitive();
}