// Vertex shader
// The output of a vertex shader is the position of the vertex
// after any kind of elaboration
// #version 430 core
#define PI 3.1415926538

layout (location = 0) in vec3 position;

layout (std430, binding = 0) buffer Positions
{
	vec2 positions[];
};

layout (std430, binding = 1) buffer Velocities
{
	vec2 velocities[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
	vec2 vel_norm = normalize(velocities[gl_InstanceID]);
	vec2 ref      = { 1,0 };
	float angle = (PI/2) - acos(clamp(dot(vel_norm, ref), -1.f, 1.f));
	mat4 rotation = mat4(
	   vec4( cos(angle), -sin(angle), 0.0,  0.0 ),
	   vec4( sin(angle), cos(angle),  0.0,  0.0 ),
	   vec4( 0.0,        0.0,         1.0,  0.0 ),
	   vec4( 0.0,        0.0,         0.0,  1.0 ) ); 
	
	mat4 instance_matrix   = mat4(1);
	vec4 instance_position = ((rotation * instance_matrix) * vec4(position, 1)) + vec4(positions[gl_InstanceID], 0, 0);
	
	gl_Position = projection_matrix * view_matrix * instance_position;
}