#include "GL/glew.h"
#include "GL/freeglut.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "sqlite3.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <SFML/Audio.hpp>

#pragma comment(lib, "freeglut")
#pragma comment(lib, "glew32")
#pragma comment(lib, "sfml-audio")

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <random>
#include <filesystem>
#include <deque>
#include <cstring>
using namespace std;
namespace fs = std::filesystem;




std::mt19937 generator_real(static_cast<unsigned>(0));
std::uniform_real_distribution<float> dis_real(0, 1);


sf::SoundBuffer explosion_buffer("media/sound/explosion.wav"); // Throws sf::Exception if an error occurs
sf::Sound explosion_sound(explosion_buffer);

sf::SoundBuffer laser_buffer("media/sound/Primary Laser.wav"); // Throws sf::Exception if an error occurs
sf::Sound laser_sound(laser_buffer);

sf::Music ms_music("media/sound/Moonlight Sonata Remix.wav");

bool draw_time_lines = false;

bool red_mode = true;

float GLOBAL_TIME = 0;
const float FPS = 30;
float DT = 1.0f / FPS;
const int COLLISION_INTERVAL_MS = 100; // 100ms = 10 times per second

// Simulation parameters
const int SIM_WIDTH = 1920;
const int SIM_HEIGHT = 1080;
const int JACOBI_ITERATIONS = 20;
const float DENSITY_DISSIPATION = 0.9f;
const float VELOCITY_DISSIPATION = 0.95f;
const float VORTICITY_SCALE = 0.1f;

float TURBULENCE_AMPLITUDE = 2.0f;      // Controls noise strength
float TURBULENCE_FREQUENCY = 10.0f;      // Controls noise frequency (scale)
float TURBULENCE_SCALE = 0.05f;          // Overall turbulence strength

float alpha = 0.5f;


bool spacePressed = false;

const float PROTAGONIST_MIN_BULLET_INTERVAL = 0.5f;


// Add a variable to track the time of the last fired bullet
std::chrono::high_resolution_clock::time_point lastBulletTime = std::chrono::high_resolution_clock::now();

bool sinusoidal_fire = false;
bool x3_fire = false;
bool x5_fire = false;


// Window dimensions
int windowWidth = 1920;
int windowHeight = 1080;

// Mouse state
int mouseX = 0, mouseY = 0;
int lastMouseX = 0, lastMouseY = 0;
bool leftMouseDown = false;
bool rightMouseDown = false;
bool middleMouseDown = false;
bool shiftDown = false;

bool upKeyPressed = false;
bool downKeyPressed = false;
bool rightKeyPressed = false;
bool leftKeyPressed = false;

// Quad VAO
GLuint quadVAO, quadVBO;



void drawQuad() {
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void setTextureUniform(GLuint program, const char* name, int unit, GLuint texture) {
	glActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D, texture);
	glUniform1i(glGetUniformLocation(program, name), unit);
}





GLuint waveChromaticProgram;

// Wave effect parameters
const int MAX_WAVE_SOURCES = 16;          // Maximum simultaneous wave effects
float waveSpeed = 2.0f;                   // How fast the wave expands (normalized units/sec)
float waveDuration = 2.0f;                // How long each wave lasts (seconds)
float waveAberrationIntensity = 0.03f;   // RGB separation strength
float waveRingWidth = 0.15f;              // Width of the distortion ring
float waveRingFalloff = 3.0f;             // How quickly the ring edges fade

// Structure to track individual wave events
struct WaveEvent {
	float centerX;      // Normalized X position (0-1)
	float centerY;      // Normalized Y position (0-1)
	float startTime;    // When the wave started (GLOBAL_TIME)
	bool active;        // Whether this slot is in use

	WaveEvent() : centerX(0), centerY(0), startTime(-100), active(false) {}
};

// Array of wave events (circular buffer approach)
WaveEvent waveEvents[MAX_WAVE_SOURCES];
int nextWaveSlot = 0;

// Toggle for the wave effect
bool waveChromaticEnabled = true;





const char* waveChromaticFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D scene;
uniform float time;
uniform float waveSpeed;
uniform float waveDuration;
uniform float aberrationIntensity;
uniform float ringWidth;
uniform float ringFalloff;
uniform float aspectRatio;

// Wave source data: each source has (centerX, centerY, startTime, active)
// Packed as vec4 for efficiency
uniform vec4 waveSources[16];
uniform int activeWaveCount;

// Creates a ring-shaped wave distortion
// Returns: distortion strength (0-1) based on distance from expanding ring front
float calculateWaveStrength(vec2 uv, vec2 center, float radius, float ringW) {
    // Aspect-correct the distance calculation
    vec2 diff = uv - center;
    diff.x *= aspectRatio;
    float dist = length(diff);
    
    // Distance from the ring front (positive = inside ring, negative = outside)
    float ringDist = abs(dist - radius);
    
    // Smooth ring falloff - strongest at the ring edge
    float strength = 1.0 - smoothstep(0.0, ringW, ringDist);
    
    return strength;
}

// Calculate direction from wave center for radial aberration
vec2 getAberrationDirection(vec2 uv, vec2 center) {
    vec2 diff = uv - center;
    diff.x *= aspectRatio;
    float len = length(diff);
    if (len < 0.001) return vec2(0.0);
    return normalize(diff);
}

void main() {
    vec3 totalOffset = vec3(0.0);
    float totalWeight = 0.0;
    
    // Accumulate distortion from all active wave sources
    for (int i = 0; i < activeWaveCount; i++) {
        vec4 source = waveSources[i];
        vec2 center = source.xy;
        float startTime = source.z;
        float isActive = source.w;
        
        if (isActive < 0.5) continue;
        
        float elapsed = time - startTime;
        if (elapsed < 0.0 || elapsed > waveDuration) continue;
        
        // Current radius of the expanding wave
        float radius = elapsed * waveSpeed;
        
        // Age-based fade out (wave weakens as it expands)
        float ageFade = 1.0 - (elapsed / waveDuration);
        ageFade = ageFade * ageFade; // Ease out
        
        // Distance-based fade (wave weakens as radius grows)
        float distFade = 1.0 / (1.0 + radius * 2.0);
        
        // Combined fade factor
        float fade = ageFade * distFade;
        
        // Ring distortion strength at this pixel
        float waveStrength = calculateWaveStrength(texCoord, center, radius, ringWidth);
        
        // Direction for radial aberration (RGB separates radially from wave center)
        vec2 aberrationDir = getAberrationDirection(texCoord, center);
        
        // Accumulate weighted distortion
        float weight = waveStrength * fade;
        
        // Create oscillating wave pattern within the ring
        float wave = sin(elapsed * 20.0 - radius * 30.0) * 0.5 + 0.5;
        weight *= mix(0.7, 1.0, wave);
        
        totalOffset.x += aberrationDir.x * weight;
        totalOffset.y += aberrationDir.y * weight;
        totalWeight += weight;
    }
    
    // If no active waves affect this pixel, just output the original
    if (totalWeight < 0.001) {
        fragColor = texture(scene, texCoord);
        return;
    }
    
    // Normalize and apply aberration intensity
    vec2 avgDir = vec2(totalOffset.x, totalOffset.y) / max(totalWeight, 0.001);
    float intensity = aberrationIntensity * min(totalWeight, 1.0);
    
    // Sample RGB channels with different offsets (radial separation)
    vec2 redOffset = avgDir * intensity * 1.2;
    vec2 greenOffset = vec2(0.0);  // Green channel stays centered
    vec2 blueOffset = -avgDir * intensity * 1.2;
    
    float r = texture(scene, texCoord + redOffset).r;
    float g = texture(scene, texCoord + greenOffset).g;
    float b = texture(scene, texCoord + blueOffset).b;
    
    // Add subtle brightness boost at wave fronts
    float brightness = 1.0 + totalWeight * 0.5;
    
    fragColor = vec4(vec3(r, g, b) * brightness, 1.0);
}
)";










float foreground_vel = -50.0f;
// Accumulated editor-only scroll applied to path_points since the last load.
// Gameplay drift is already accounted for elsewhere (the spline gets consumed
// as the enemy progresses), but editor arrow-key scroll bakes a raw offset
// into every path_points[].x with no counterpart in the persisted state.
// editorSaveToDatabase subtracts this so canonical positions round-trip.
// Reset to 0 on load and on save (after using it).
float g_editorScrollAccum = 0.0f;
float g_loadTimeFgX = 0.0f;



// Helper: Evaluate a single Catmull-Rom segment given 4 control points and t in [0,1]
float catmull_rom_segment(float p0, float p1, float p2, float p3, float t)
{
	float t2 = t * t;
	float t3 = t2 * t;

	// Catmull-Rom basis matrix coefficients
	return 0.5f * (
		(2.0f * p1) +
		(-p0 + p2) * t +
		(2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
		(-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
		);
}

glm::vec2 catmull_rom_segment(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec2 p3, float t)
{
	glm::vec2 result;
	result.x = catmull_rom_segment(p0.x, p1.x, p2.x, p3.x, t);
	result.y = catmull_rom_segment(p0.y, p1.y, p2.y, p3.y, t);
	return result;
}


// Get a point on a Catmull-Rom spline
// points: control points (minimum 2)
// t: parameter in [0, 1] spanning the entire curve
// The curve passes through all control points
float get_spline_point(const vector<float>& points, float t)
{
	if (points.size() == 0)
		return 0.0f;

	if (points.size() == 1)
		return points[0];

	if (points.size() == 2)
		return points[0] + t * (points[1] - points[0]); // Linear interpolation

	// Clamp t to [0, 1]
	if (t <= 0.0f) return points[0];
	if (t >= 1.0f) return points[points.size() - 1];

	// Number of segments is (n - 1) for n points
	size_t num_segments = points.size() - 1;

	// Find which segment we're in
	float scaled_t = t * num_segments;
	size_t segment = static_cast<size_t>(scaled_t);

	// Handle edge case where t == 1.0
	if (segment >= num_segments)
		segment = num_segments - 1;

	// Local t within the segment [0, 1]
	float local_t = scaled_t - segment;

	// Get the 4 control points for this segment
	// For endpoints, we extrapolate phantom points
	float p0, p1, p2, p3;

	p1 = points[segment];
	p2 = points[segment + 1];

	// Handle p0 (point before p1)
	if (segment == 0)
		p0 = 2.0f * p1 - p2; // Extrapolate: reflect p2 across p1
	else
		p0 = points[segment - 1];

	// Handle p3 (point after p2)
	if (segment + 2 >= points.size())
		p3 = 2.0f * p2 - p1; // Extrapolate: reflect p1 across p2
	else
		p3 = points[segment + 2];

	return catmull_rom_segment(p0, p1, p2, p3, local_t);
}


glm::vec2 get_spline_point(const vector<glm::vec2>& points, float t)
{
	glm::vec2 result;
	result.x = result.y = 0.0f;

	if (points.size() == 0)
		return result;

	if (points.size() == 1)
		return points[0];

	if (points.size() == 2)
	{
		result.x = points[0].x + t * (points[1].x - points[0].x);
		result.y = points[0].y + t * (points[1].y - points[0].y);
		return result;
	}

	// Clamp t to [0, 1]
	if (t <= 0.0f) return points[0];
	if (t >= 1.0f) return points[points.size() - 1];

	// Number of segments is (n - 1) for n points
	size_t num_segments = points.size() - 1;

	// Find which segment we're in
	float scaled_t = t * num_segments;
	size_t segment = static_cast<size_t>(scaled_t);

	// Handle edge case where t == 1.0
	if (segment >= num_segments)
		segment = num_segments - 1;

	// Local t within the segment [0, 1]
	float local_t = scaled_t - segment;

	// Get the 4 control points for this segment
	glm::vec2 p0, p1, p2, p3;

	p1 = points[segment];
	p2 = points[segment + 1];

	// Handle p0 (point before p1)
	if (segment == 0)
	{
		p0.x = 2.0f * p1.x - p2.x;
		p0.y = 2.0f * p1.y - p2.y;
	}
	else
	{
		p0 = points[segment - 1];
	}

	// Handle p3 (point after p2)
	if (segment + 2 >= points.size())
	{
		p3.x = 2.0f * p2.x - p1.x;
		p3.y = 2.0f * p2.y - p1.y;
	}
	else
	{
		p3 = points[segment + 2];
	}

	return catmull_rom_segment(p0, p1, p2, p3, local_t);
}


// Optional: Get the tangent (derivative) at a point on the spline
// Useful for orienting objects along the path
glm::vec2 get_spline_tangent(const vector<glm::vec2>& points, float t)
{
	glm::vec2 result;
	result.x = result.y = 0.0f;

	if (points.size() < 2)
		return result;

	if (points.size() == 2)
	{
		result.x = points[1].x - points[0].x;
		result.y = points[1].y - points[0].y;
		return result;
	}

	// Clamp t
	t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;

	size_t num_segments = points.size() - 1;
	float scaled_t = t * num_segments;
	size_t segment = static_cast<size_t>(scaled_t);

	if (segment >= num_segments)
		segment = num_segments - 1;

	float local_t = scaled_t - segment;

	glm::vec2 p0, p1, p2, p3;
	p1 = points[segment];
	p2 = points[segment + 1];

	if (segment == 0)
	{
		p0.x = 2.0f * p1.x - p2.x;
		p0.y = 2.0f * p1.y - p2.y;
	}
	else
	{
		p0 = points[segment - 1];
	}

	if (segment + 2 >= points.size())
	{
		p3.x = 2.0f * p2.x - p1.x;
		p3.y = 2.0f * p2.y - p1.y;
	}
	else
	{
		p3 = points[segment + 2];
	}

	// Derivative of Catmull-Rom
	float t2 = local_t * local_t;

	result.x = 0.5f * (
		(-p0.x + p2.x) +
		2.0f * (2.0f * p0.x - 5.0f * p1.x + 4.0f * p2.x - p3.x) * local_t +
		3.0f * (-p0.x + 3.0f * p1.x - 3.0f * p2.x + p3.x) * t2
		);

	result.y = 0.5f * (
		(-p0.y + p2.y) +
		2.0f * (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * local_t +
		3.0f * (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t2
		);

	return result;
}













// Pre-calculates the actual time (in seconds) for an enemy to traverse the
// entire spline path, accounting for the variable speed profile in path_speeds.
//
// With a uniform speed of 1.0 everywhere the result equals path_animation_length.
// Speeds < 1 lengthen the traversal proportionally; speeds > 1 shorten it.
//
// The integral  T = path_animation_length * integral_0^1 (1/speed(t)) dt
// is evaluated numerically with the midpoint rule.
//
// path_points is included in the signature so that an arc-length-weighted
// variant can be added here in the future without changing all call sites.
float calculate_actual_path_duration(
	const std::vector<glm::vec2>& /*path_points*/,
	const std::vector<float>& path_speeds,
	float path_animation_length)
{
	if (path_speeds.empty() || path_animation_length <= 0.0f)
		return path_animation_length;

	const int N = 512;
	float inv_speed_sum = 0.0f;
	for (int i = 0; i < N; ++i)
	{
		float t = (i + 0.5f) / static_cast<float>(N);
		float speed = get_spline_point(path_speeds, t);
		if (speed > 0.0f)
			inv_speed_sum += 1.0f / speed;
	}
	// Average of 1/speed over [0,1] scaled by path_animation_length gives
	// the true traversal duration regardless of the speed profile.
	return path_animation_length * (inv_speed_sum / static_cast<float>(N));
}


float get_curve_point(vector<float> points, float t)
{
	if (points.size() == 0)
		return 0;

	size_t i = points.size() - 1;

	while (i > 0)
	{
		for (int k = 0; k < i; k++)
			points[k] += t * (points[k + 1] - points[k]);

		i--;
	}

	return points[0];
}


glm::vec2 get_curve_point(vector<glm::vec2> points, float t)
{
	if (points.size() == 0)
	{
		glm::vec2 vd;
		vd.x = vd.y = 0;
		return vd;
	}

	size_t i = points.size() - 1;

	while (i > 0)
	{
		for (int k = 0; k < i; k++)
		{
			points[k].x += t * (points[k + 1].x - points[k].x);
			points[k].y += t * (points[k + 1].y - points[k].y);
		}

		i--;
	}

	return points[0];
}






struct CompareVec2
{
	bool operator()(const glm::vec2& lhs, const glm::vec2& rhs) const
	{
		if (lhs.x != rhs.x)
		{
			return lhs.x < rhs.x;
		}

		return lhs.y < rhs.y;
	}
};


void RandomUnitVector(float& x_out, float& y_out)
{
	const static float pi = 4.0f * atanf(1.0f);

	const float a = dis_real(generator_real) * 2.0f * pi;

	x_out = cos(a);
	y_out = sin(a);
}

float naive_lerp(float a, float b, float t)
{
	return a + t * (b - a);
}

glm::vec3 hsbToRgb(float hue, float saturation, float brightness) {
	// hue: 0-360, saturation: 0-1, brightness: 0-1

	float c = brightness * saturation;
	float x = c * (1 - std::fabsf(std::fmodf(hue / 60.0f, 2) - 1));
	float m = brightness - c;

	float r, g, b;

	if (hue < 60) {
		r = c; g = x; b = 0;
	}
	else if (hue < 120) {
		r = x; g = c; b = 0;
	}
	else if (hue < 180) {
		r = 0; g = c; b = x;
	}
	else if (hue < 240) {
		r = 0; g = x; b = c;
	}
	else if (hue < 300) {
		r = x; g = 0; b = c;
	}
	else {
		r = c; g = 0; b = x;
	}

	int red = static_cast<int>((r + m) * 255);
	int green = static_cast<int>((g + m) * 255);
	int blue = static_cast<int>((b + m) * 255);

	return { red, green, blue };
}

struct HashVec2
{
	size_t operator()(const glm::vec2& v) const
	{
		size_t h1 = std::hash<float>{}(v.x);
		size_t h2 = std::hash<float>{}(v.y);
		return h1 ^ (h2 << 1);
	}
};

struct EqualVec2
{
	bool operator()(const glm::vec2& lhs, const glm::vec2& rhs) const
	{
		return lhs.x == rhs.x && lhs.y == rhs.y;
	}
};



class pre_sprite
{
public:

	GLuint tex = 0;

	int width = 0;
	int height = 0;
	float x = 0;
	float y = 0;
	float old_x = 0;
	float old_y = 0;
	float vel_x = 0;
	float vel_y = 0;

	bool under_fire = false;



	// unordered_map<glm::vec2, float, HashVec2, EqualVec2> blackening_age_map;
	map<glm::vec2, float, CompareVec2> blackening_age_map;

	vector<unsigned char*> to_present_data_pointers;

	virtual void update_tex(void) = 0;

	bool to_be_culled = false;

	bool isOnscreen(void) const
	{
		return
			(x + width > 0) &&
			(x < windowWidth) &&
			(y + height > 0) &&
			(y < windowHeight);
	}

	void integrate(float dt)
	{
		if (to_be_culled)
			return;

		old_x = x;
		old_y = y;

		x = x + vel_x * dt;
		y = y + vel_y * dt;
	}


	void animate_blackening(const vector<glm::vec2>& locations, size_t state)
	{
		const float glut_curr_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
		const float BRUSH_RADIUS = 10.0f;        // Radius of the soft brush in sprite pixels
		const float BRUSH_RADIUS_SQUARED = BRUSH_RADIUS * BRUSH_RADIUS;
		const float transparent_threshold = 0.999f;
		const float animation_length = 5.0f;

		for (size_t i = 0; i < locations.size(); i++)
			blackening_age_map[locations[i]] = glut_curr_time;

		// Remove blackening_age_map entries where the circular brush area is now fully transparent
		for (map<glm::vec2, float>::iterator it = blackening_age_map.begin(); it != blackening_age_map.end(); )
		{
			glm::vec2 centre = it->first;

			int minX = std::max(0, (int)(centre.x - BRUSH_RADIUS - 1));
			int maxX = std::min(width - 1, (int)(centre.x + BRUSH_RADIUS + 1));
			int minY = std::max(0, (int)(centre.y - BRUSH_RADIUS - 1));
			int maxY = std::min(height - 1, (int)(centre.y + BRUSH_RADIUS + 1));

			bool all_transparent = true;

			for (int y = minY; y <= maxY && all_transparent; ++y)
			{
				for (int x = minX; x <= maxX && all_transparent; ++x)
				{
					glm::vec2 diff(x - centre.x, y - centre.y);
					float distSq = diff.x * diff.x + diff.y * diff.y;

					if (distSq < BRUSH_RADIUS_SQUARED)
					{
						const size_t index = (y * width + x) * 4 + 3;

						if (to_present_data_pointers[state][index] != 0)
							all_transparent = false;
					}
				}
			}

			if (all_transparent)
			{
				it = blackening_age_map.erase(it);
				continue;
			}

			++it;
		}








		bool transparent = false;

		for (map<glm::vec2, float>::const_iterator ci = blackening_age_map.begin(); ci != blackening_age_map.end(); ci++)
		{
			glm::vec2 point(ci->first.x, ci->first.y);

			int minX = std::max(0, (int)(point.x - BRUSH_RADIUS - 1));
			int maxX = std::min(width - 1, (int)(point.x + BRUSH_RADIUS + 1));
			int minY = std::max(0, (int)(point.y - BRUSH_RADIUS - 1));
			int maxY = std::min(height - 1, (int)(point.y + BRUSH_RADIUS + 1));



			// Do erosion
			if (dis_real(generator_real) > transparent_threshold)
				transparent = true;

			for (int y = minY; y <= maxY; ++y)
			{
				for (int x = minX; x <= maxX; ++x)
				{
					glm::vec2 diff(x - point.x, y - point.y);
					float distSq = diff.x * diff.x + diff.y * diff.y;

					if (distSq < BRUSH_RADIUS_SQUARED)
					{
						const size_t index = (y * width + x) * 4;

						const float duration = glut_curr_time - ci->second;



						if (duration >= animation_length)
						{
							to_present_data_pointers[state][index + 0] = 0;
							to_present_data_pointers[state][index + 1] = 0;
							to_present_data_pointers[state][index + 2] = 0;
						}
						else
						{
							//glm::vec3 red_colour = hsbToRgb(60 - 60 * duration / animation_length, duration / animation_length, powf(1.0f - duration / animation_length, 0.25));

							// From JoeJ on gamedev.net
							float t = 1 - duration / animation_length;
							float t2 = t * t;
							float t4 = t2 * t2;
							float r = 255.0f * min(1.f, t * 1.8f);
							float g = 255.0f * min(1.f, t2 * 1.5f);
							float b = 255.0f * t4;

							to_present_data_pointers[state][index + 0] = static_cast<unsigned int>(r);
							to_present_data_pointers[state][index + 1] = static_cast<unsigned int>(g);
							to_present_data_pointers[state][index + 2] = static_cast<unsigned int>(b);

							if (transparent && duration / animation_length < 0.001)
							{
								to_present_data_pointers[state][index + 3] = 0;
								continue;
							}
						}
					}
				}
			}
		}
















		for (size_t i = 0; i < to_present_data_pointers.size(); i++)
		{
			if (to_present_data_pointers[i] == 0)
				continue;

			if (i == state)
			{
				continue;
			}

			for (map<glm::vec2, float>::const_iterator ci = blackening_age_map.begin(); ci != blackening_age_map.end(); ci++)
			{
				glm::vec2 point(ci->first.x, ci->first.y);

				// Transform point from current state to target state i.
				// For the column at point.x, find the first and last non-transparent
				// rows in both the source state and target state. These define a
				// vertical gradient (0 at top, 1 at bottom). The point's relative
				// position within the source extent maps to the same relative
				// position in the target extent.
				//
				// To avoid drift from erosion (which sets alpha=0 and shrinks the
				// detected extent over time), we also skip target pixels that are
				// already fully transparent -- only paint onto pixels that still
				// have substance.

				int col = (int)(point.x + 0.5f);
				if (col < 0) col = 0;
				if (col >= width) col = width - 1;

				// Find first and last non-transparent rows in the source (current) state
				int src_first = -1, src_last = -1;
				for (int row = 0; row < height; ++row)
				{
					unsigned char alpha = to_present_data_pointers[state][(row * width + col) * 4 + 3];
					if (alpha > 0)
					{
						if (src_first == -1) src_first = row;
						src_last = row;
					}
				}

				// Find first and last non-transparent rows in the target state i,
				// but also consider blackened (RGB=0) pixels with alpha>0 as valid
				// -- only truly transparent (alpha==0) pixels that were ORIGINALLY
				// transparent are excluded. We detect original transparency by
				// checking if the pixel is alpha==0 AND rgb==0 (eroded) vs
				// alpha==0 from the original image. Since eroded pixels had their
				// alpha set to 0 by this code, and original transparent pixels also
				// have alpha==0, we can't distinguish them after the fact.
				//
				// Instead, use the source state's extent as the stable reference
				// for the target too. The source extent is where the hit actually
				// landed, so it's the most reliable anchor.
				int dst_first = src_first;
				int dst_last = src_last;

				// Skip if source has no visible pixels in this column
				if (src_first == -1)
					continue;

				// Compute normalized position of point.y within the source extent
				float t_norm = 0.5f; // default to midpoint if source extent is zero-height
				if (src_last > src_first)
					t_norm = (point.y - (float)src_first) / (float)(src_last - src_first);

				// Apply cosine-based mapping to account for rotation.
				// Linear t_norm assumes uniform vertical stretching, but rotation
				// compresses/expands with a cosine profile (center stays put,
				// edges move most). This S-curve approximates that.
				//float cos_t = 0.5f * (1.0f - cosf(t_norm * 3.14159265f));

				// Map to target extent using the cosine-weighted parameter
				//float mapped_y = (float)dst_first + cos_t * (float)(dst_last - dst_first);
				float mapped_y = (float)dst_first + t_norm * (float)(dst_last - dst_first);
				glm::vec2 mapped_point(point.x, mapped_y);

				int minX = std::max(0, (int)(mapped_point.x - BRUSH_RADIUS - 1));
				int maxX = std::min(width - 1, (int)(mapped_point.x + BRUSH_RADIUS + 1));
				int minY = std::max(0, (int)(mapped_point.y - BRUSH_RADIUS - 1));
				int maxY = std::min(height - 1, (int)(mapped_point.y + BRUSH_RADIUS + 1));

				for (int y = minY; y <= maxY; ++y)
				{
					for (int x = minX; x <= maxX; ++x)
					{
						glm::vec2 diff(x - mapped_point.x, y - mapped_point.y);
						float distSq = diff.x * diff.x + diff.y * diff.y;

						if (distSq < BRUSH_RADIUS_SQUARED)
						{
							const size_t index = (y * width + x) * 4;

							const float duration = glut_curr_time - ci->second;

							const float animation_length = 5.0;

							if (duration >= animation_length)
							{
								to_present_data_pointers[i][index + 0] = 0;
								to_present_data_pointers[i][index + 1] = 0;
								to_present_data_pointers[i][index + 2] = 0;
							}
							else
							{
								// From JoeJ on gamedev.net
								float t = 1 - duration / animation_length;
								float t2 = t * t;
								float t4 = t2 * t2;
								float r = 255.0f * min(1.f, t * 1.8f);
								float g = 255.0f * min(1.f, t2 * 1.5f);
								float b = 255.0f * t4;

								to_present_data_pointers[i][index + 0] = static_cast<unsigned int>(r);
								to_present_data_pointers[i][index + 1] = static_cast<unsigned int>(g);
								to_present_data_pointers[i][index + 2] = static_cast<unsigned int>(b);

								if (transparent && duration / animation_length < 0.001)
								{
									to_present_data_pointers[i][index + 3] = 0;
									continue;
								}
							}
						}
					}
				}
			}
		}



		update_tex();
	}





};




class sprite : public pre_sprite
{
public:

	vector<unsigned char> to_present_data;

	sprite(const sprite& other)
		: pre_sprite(other),
		to_present_data(other.to_present_data)
	{
		to_present_data_pointers.clear();
		if (!to_present_data.empty())
			to_present_data_pointers.push_back(to_present_data.data());
	}

	sprite(void)
	{
		// Leave to_present_data_pointers empty until data is loaded.
		// Whatever code populates to_present_data should also push the pointer.
	}


	void update_tex(void)
	{
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, to_present_data.data());
	}
};



const int UP_STATE = 0;
const int DOWN_STATE = 1;
const int REST_STATE = 2;

class tri_sprite : public pre_sprite
{
public:

	std::vector<std::vector<unsigned char>> sprite_frames;
	std::vector<std::vector<unsigned char>> original_sprite_frames;



	int state = 0; // will be set to rest_state_index() after frames are loaded

	int rest_state_index() const { return static_cast<int>(sprite_frames.size()) / 2; }
	int num_frames()       const { return static_cast<int>(sprite_frames.size()); }

	void rebuild_pointers()
	{
		to_present_data_pointers.clear();
		for (size_t i = 0; i < sprite_frames.size(); i++)
			to_present_data_pointers.push_back(sprite_frames[i].data());
	}

	tri_sprite(void)
	{
		// Start with 3 empty frames so legacy code that checks .empty() works
		// before real data is loaded.
		sprite_frames.resize(3);
		state = rest_state_index();
		original_sprite_frames = sprite_frames;
	}

	// Accept an arbitrary number of frames (must be odd).
	void manually_update_data(
		const std::vector<std::vector<unsigned char>>& src_frames)
	{
		// Resize our frame vector to match.
		if (sprite_frames.size() != src_frames.size())
			sprite_frames.resize(src_frames.size());

		for (size_t i = 0; i < src_frames.size(); i++)
		{
			if (!src_frames[i].empty())
				sprite_frames[i] = src_frames[i];
		}

		original_sprite_frames = sprite_frames;

		state = rest_state_index();

		// Create a unique OpenGL texture for this instance so that
		// multiple sprites cloned from the same template do not
		// overwrite each other when update_tex() is called.
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		rebuild_pointers();
		update_tex();
	}

	// Legacy 3-argument overload for call sites that still pass three vectors.
	void manually_update_data(
		const std::vector<unsigned char>& src_to_present_up_data,
		const std::vector<unsigned char>& src_to_present_down_data,
		const std::vector<unsigned char>& src_to_present_rest_data)
	{
		std::vector<std::vector<unsigned char>> frames(3);
		frames[0] = src_to_present_up_data;
		frames[1] = src_to_present_rest_data;
		frames[2] = src_to_present_down_data;
		manually_update_data(frames);
	}



	//----------------------------------------------------------------------
	//  Update OpenGL texture based on state
	//----------------------------------------------------------------------
	void update_tex()
	{
		if (sprite_frames.empty()) return;

		// Clamp state into valid range.
		int idx = state;
		if (idx < 0) idx = 0;
		if (idx >= (int)sprite_frames.size()) idx = (int)sprite_frames.size() - 1;

		glBindTexture(GL_TEXTURE_2D, tex);

		if (!sprite_frames[idx].empty())
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
				GL_RGBA, GL_UNSIGNED_BYTE,
				sprite_frames[idx].data());
		else if (!sprite_frames[rest_state_index()].empty())
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
				GL_RGBA, GL_UNSIGNED_BYTE,
				sprite_frames[rest_state_index()].data());
	}
};





class power_up : public sprite
{
public:

	// Sinusoidal wave motion state. wave_time accumulates per-instance so each
	// power-up has its own phase. Frequency is in radians/sec, amplitude in pixels.
	float wave_time = 0.0f;
	float wave_frequency = 6.0f;
	float wave_amplitude = 50.0f;

	power_up()
	{
		// Randomize the starting phase so multiple power-ups spawned from the
		// same enemy at the same instant don't trace identical paths. We seed
		// wave_time with a value in [0, 2*PI / wave_frequency), which covers a
		// full period of the sine wave.
		static const float TWO_PI = 8.0f * atan(1.0f);
		wave_time = dis_real(generator_real) * (TWO_PI / wave_frequency);
		wave_amplitude = 50.0f + dis_real(generator_real) * 50.0f;
		wave_frequency = 6.0f;
	}

	virtual void integrate(float dt)
	{
		old_x = x;
		old_y = y;

		// Compute the unit direction of travel from the velocity vector.
		float dirX = vel_x;
		float dirY = vel_y;
		float dirLength = sqrt(dirX * dirX + dirY * dirY);
		if (dirLength > 0.0f) {
			dirX /= dirLength;
			dirY /= dirLength;
		}

		// Move forward along original path
		x += vel_x * dt;
		y += vel_y * dt;

		// Apply sinusoidal motion perpendicular to the direction of travel.
		// We add the *delta* of the sine wave between this frame and the last,
		// so the perpendicular offset accumulates correctly on top of forward
		// motion without snapping back to a baseline each frame.
		float perpX = -dirY;
		float perpY = dirX;

		float prevSin = sin(wave_time * wave_frequency);
		wave_time += dt;
		float currSin = sin(wave_time * wave_frequency);
		float waveDelta = (currSin - prevSin) * wave_amplitude;

		x += perpX * waveDelta;
		y += perpY * waveDelta;
	}


};

class sinusoidal_fire_power_up : public power_up
{
public:
};

class x3_fire_power_up : public power_up
{
public:
};

class x5_fire_power_up : public power_up
{
public:
};





class ship : public tri_sprite
{
public:

	float health;
	float max_health;

	ship() : health(1.0), max_health(1.0) {}
};



class friendly_ship : public ship {
public:
	float last_time_collided = 0;

	// Tilt animation state
	float current_tilt = 0.0f;      // -1.0 = full up, +1.0 = full down, 0 = rest
	float target_tilt = 0.0f;       // What the player is currently trying to do

	float tilt_speed = 3.0f;        // How fast we approach target (higher = snappier)

	friendly_ship() : ship() {
		health = 1000.0f;
		max_health = 1000.0f;
	}


	void updateTiltFromInput()
	{
		if (upKeyPressed) {
			target_tilt = -1.0f;
		}
		else if (downKeyPressed) {
			target_tilt = 1.0f;
		}
		else {
			target_tilt = 0.0f;
		}

		updateTilt();
	}


	// Target velocity set by key input; actual vel_x/vel_y lerps toward this
	float target_vel_x = 0.0f;
	float target_vel_y = 0.0f;

	// How quickly velocity ramps up when a key is held (units/sec²)
	float acceleration = 8.0f;
	// How quickly velocity fades when no key is held (drag multiplier, 0-1 per second)
	float drag = 6.0f;

	void set_velocity(const float src_x, const float src_y)
	{
		vel_x = src_x;
		vel_y = src_y;
	}

	// Sets the *desired* velocity from key input; inertia is applied in applyInertia()
	void set_target_velocity(const float src_x, const float src_y)
	{
		target_vel_x = src_x;
		target_vel_y = src_y;
	}

	// Called once per frame to smoothly blend actual velocity toward target
	void applyInertia(float dt)
	{
		auto lerp_axis = [&](float current, float target, float acc, float drg) -> float {
			if (target != 0.0f) {
				// Accelerate toward target velocity
				float diff = target - current;
				float step = acc * std::abs(target) * dt;
				if (std::abs(diff) <= step)
					return target;
				return current + std::copysign(step, diff);
			}
			else {
				// No key held — apply drag to fade out
				float step = drg * std::abs(current) * dt;
				if (std::abs(current) <= step)
					return 0.0f;
				return current - std::copysign(step, current);
			}
			};

		vel_x = lerp_axis(vel_x, target_vel_x, acceleration, drag);
		vel_y = lerp_axis(vel_y, target_vel_y, acceleration, drag);
	}


	void updateTilt()
	{
		const int n = num_frames();
		if (n < 3)
		{
			state = rest_state_index();
			update_tex();
			return;
		}

		// Move toward target tilt at a constant linear rate
		float diff = target_tilt - current_tilt;
		float max_step = tilt_speed * DT;

		if (std::abs(diff) <= max_step)
			current_tilt = target_tilt;
		else
			current_tilt += std::copysign(max_step, diff);

		current_tilt = std::clamp(current_tilt, -1.0f, 1.0f);

		// Map tilt [-1, 1] to frame [0, n-1]
		float normalized = (current_tilt + 1.0f) / 2.0f;
		int target_frame = std::clamp((int)((n - 1) * normalized), 0, n - 1);

		state = target_frame;
		update_tex();
	}


};



#define CANNON_TYPE_LEFT 0
#define CANNON_TYPE_UP_DOWN 1
#define CANNON_TYPE_TRACKING 2
#define CANNON_TYPE_CIRCULAR 3

// Power-up types. IDs are 0-based in memory; DB stores them 1-based
// (see power_up seed rows in ensureDatabaseSchema).
#define POWER_UP_TYPE_SINUSOIDAL 0
#define POWER_UP_TYPE_X3         1
#define POWER_UP_TYPE_X5         2
#define NUM_POWER_UP_TYPES       3


class cannon
{
public:
	double min_bullet_interval = 0;
	int cannon_type = 0;
	double x = 0;
	double y = 0;
	std::chrono::high_resolution_clock::time_point lastBulletTime = std::chrono::high_resolution_clock::now();
};

// Enemy sprites are sliced into a grid of chunks (mirroring how the
// foreground is chunked). Each chunk is a small multi-frame tri_sprite so
// that tilt frames and animate_blackening (including cross-frame
// propagation and erosion-based transparency) continue to work per-chunk.
const int enemy_chunk_size_width = 128;
const int enemy_chunk_size_height = 128;

class enemy_chunk : public tri_sprite
{
public:
	// Offset (top-left, in sprite-local pixel coords) of this chunk within
	// the parent enemy sprite. Used both for drawing (enemy.x + offset_x)
	// and for converting enemy-local hit coords into chunk-local ones.
	int offset_x = 0;
	int offset_y = 0;
};

class enemy_ship : public ship
{
public:

	//enemy_ship() : ship()
	//{
	//	health = 50.0f;
	//	max_health = 50.0f;
	//}

	int template_idx = 0; // tracks which enemy_template this ship is currently using

	// Throttle blackening to at most once every 100ms per enemy.
	// last_blacken_time_ms is the timestamp (in ms since epoch) of the
	// last accepted blackenChunks call; BLACKEN_INTERVAL_MS is the
	// minimum spacing between accepted calls. Tracked per-instance so
	// each enemy has an independent cooldown.
	long long last_blacken_time_ms = 0;
	static const long long BLACKEN_INTERVAL_MS = 100;

	// Visual decomposition of this enemy into a grid of small chunks.
	// Populated by rebuildChunks() after manually_update_data. Templates
	// leave this empty; only live enemies (in enemy_ships) build chunks.
	vector<enemy_chunk> chunks;

	// Build chunks from the current sprite_frames. Slices each frame into
	// enemy_chunk_size_width x enemy_chunk_size_height tiles. Chunks whose
	// every frame is fully transparent are skipped.
	void rebuildChunks()
	{
		// Release any GL textures owned by a previous chunk set before
		// dropping the chunks vector.
		for (size_t i = 0; i < chunks.size(); ++i)
		{
			if (chunks[i].tex != 0)
				glDeleteTextures(1, &chunks[i].tex);
		}
		chunks.clear();

		if (sprite_frames.empty() || width <= 0 || height <= 0)
			return;

		const int cw = enemy_chunk_size_width;
		const int ch = enemy_chunk_size_height;
		const int tilesX = (width + cw - 1) / cw;
		const int tilesY = (height + ch - 1) / ch;
		const int nFrames = static_cast<int>(sprite_frames.size());

		chunks.reserve(tilesX * tilesY);

		for (int ty = 0; ty < tilesY; ++ty)
		{
			for (int tx = 0; tx < tilesX; ++tx)
			{
				const int srcStartX = tx * cw;
				const int srcStartY = ty * ch;
				const int copyW = std::min(cw, width - srcStartX);
				const int copyH = std::min(ch, height - srcStartY);

				// Build a frame-sliced copy for this chunk. Each chunk
				// frame is a full cw x ch buffer (padded transparent on
				// the edges) so that animate_blackening's brush math
				// works with uniform dimensions.
				std::vector<std::vector<unsigned char>> chunk_frames(nFrames);
				bool any_opaque = false;

				for (int f = 0; f < nFrames; ++f)
				{
					chunk_frames[f].assign(cw * ch * 4, 0);

					if (sprite_frames[f].empty())
						continue;

					const unsigned char* src = sprite_frames[f].data();

					for (int y = 0; y < copyH; ++y)
					{
						for (int x = 0; x < copyW; ++x)
						{
							const int srcIdx = ((srcStartY + y) * width + (srcStartX + x)) * 4;
							const int dstIdx = (y * cw + x) * 4;
							chunk_frames[f][dstIdx + 0] = src[srcIdx + 0];
							chunk_frames[f][dstIdx + 1] = src[srcIdx + 1];
							chunk_frames[f][dstIdx + 2] = src[srcIdx + 2];
							chunk_frames[f][dstIdx + 3] = src[srcIdx + 3];
							if (src[srcIdx + 3] != 0)
								any_opaque = true;
						}
					}
				}

				// Skip fully-transparent chunks -- nothing to draw or
				// blacken.
				if (!any_opaque)
					continue;

				enemy_chunk c;
				c.offset_x = srcStartX;
				c.offset_y = srcStartY;
				c.width = cw;
				c.height = ch;
				// manually_update_data allocates a unique GL texture
				// and populates to_present_data_pointers for all frames.
				c.manually_update_data(chunk_frames);
				c.state = state;  // mirror parent's current frame

				chunks.push_back(std::move(c));
				// After the move the inner vector buffers have been
				// transferred, but to_present_data_pointers was cached
				// against the old object's addresses. Rebuild it so it
				// unambiguously points into the new chunk's sprite_frames.
				chunks.back().rebuild_pointers();
			}
		}
	}

	float appearance_time = 0;
	float path_animation_length = 0; // seconds
	vector<glm::vec2> path_points;
	vector<float> path_speeds;
	vector<cannon> cannons;

	// List of power-up types owned by this enemy. Each entry is a
	// POWER_UP_TYPE_* constant (0-based). Persisted via the
	// enemy_power_up table.
	vector<int> power_ups;

	int path_pixel_delay = 0;
	float path_scroll_rate = 0.0f; // computed at activation

	// Accumulated foreground drift (in pixels) that occurred while this
	// enemy was in the active spline phase, where path_points are held
	// still in screen space but the foreground keeps drifting at
	// foreground_vel. editorSaveToDatabase adds this back into the
	// canonicalisation formula so positions round-trip correctly.
	// Sign matches foreground_vel * elapsed (i.e. negative when drifting
	// left). Reset on load and on save.
	float spline_phase_drift = 0.0f;

	float path_t = -1.0f;


	float current_tilt = 0.0f;      // -1.0 = full up, +1.0 = full down, 0 = rest
	float target_tilt = 0.0f;       // What the player is currently trying to do

	float tilt_speed = 1.0f;        // How fast we approach target (higher = snappier)

	void updateTiltFromInput()
	{
		if (vel_y < 0.1)
			target_tilt = -1.0f;
		else if (vel_y > 0.1)
			target_tilt = 1.0f;
		else
			target_tilt = 0.0f;

		updateTilt();
	}


	void set_velocity(const float src_x, const float src_y)
	{
		vel_x = src_x;
		vel_y = src_y;

		// Determine desired tilt
		if (src_y < -1.0f) {           // Moving up
			target_tilt = -1.0f;
		}
		else if (src_y > 1.0f) {       // Moving down
			target_tilt = 1.0f;
		}
		else {
			target_tilt = 0.0f;         // No vertical input → return to center
		}

		updateTilt();
	}


	void updateTilt()
	{
		const int n = num_frames();
		if (n < 3)
		{
			state = rest_state_index();
			update_tex();
			return;
		}

		// Move toward target tilt at a constant linear rate
		float diff = target_tilt - current_tilt;
		float max_step = tilt_speed * DT;

		if (std::abs(diff) <= max_step)
			current_tilt = target_tilt;
		else
			current_tilt += std::copysign(max_step, diff);

		current_tilt = std::clamp(current_tilt, -1.0f, 1.0f);

		// Map tilt [-1, 1] to frame [0, n-1]
		float normalized = (current_tilt + 1.0f) / 2.0f;
		int target_frame = std::clamp((int)((n - 1) * normalized), 0, n - 1);

		state = target_frame;
		update_tex();
	}



	void integrate(float dt)
	{
		if (to_be_culled)
			return;

		old_x = x;
		old_y = y;

		x = x + vel_x * dt;
		y = y + vel_y * dt;
	}

	// Route a batch of enemy-local hit points to the appropriate chunks.
	// Each chunk receives only the hits that land inside it (converted to
	// chunk-local coords), and its own animate_blackening handles
	// coloring, cross-frame propagation, and erosion-based transparency.
	// Chunks that receive no hits this frame still need to advance their
	// blackening animation (so existing damage keeps progressing toward
	// fully black / eroded), so we call animate_blackening with an empty
	// list on them too.
	void blackenChunks(const vector<glm::vec2>& enemy_local_hits)
	{
		if (chunks.empty())
			return;

		// Throttle: skip if less than 100ms has passed since the last
		// blackening on this specific enemy.
		auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		if (now_ms - last_blacken_time_ms < BLACKEN_INTERVAL_MS)
			return;
		last_blacken_time_ms = now_ms;

		for (size_t ci = 0; ci < chunks.size(); ++ci)
		{
			enemy_chunk& c = chunks[ci];

			vector<glm::vec2> local_hits;
			for (size_t i = 0; i < enemy_local_hits.size(); ++i)
			{
				const float lx = enemy_local_hits[i].x - (float)c.offset_x;
				const float ly = enemy_local_hits[i].y - (float)c.offset_y;
				if (lx >= 0 && lx < (float)c.width &&
					ly >= 0 && ly < (float)c.height)
				{
					local_hits.push_back(glm::vec2(lx, ly));
				}
			}

			// Keep each chunk's displayed frame in sync with the parent
			// enemy's current tilt state so blackening paints onto the
			// frame the player actually sees.
			c.state = state;
			c.animate_blackening(local_hits, c.state);
		}

		// animate_blackening mutates each chunk's pixel buffers (coloring,
		// fully-black pixels, and erosion-driven alpha=0). The parent
		// enemy's own sprite_frames are still the ORIGINAL un-damaged
		// pixels -- but they're what isPixelInsideTriSpriteAndTransparent
		// and getTriSpriteActiveData read when deciding whether a bullet
		// hit a solid pixel. If we don't propagate the chunk edits back,
		// bullets keep registering hits on pixels the player sees as
		// already eroded away.
		//
		// We sync every frame (not just the currently displayed one)
		// because animate_blackening's cross-state propagation writes
		// into all other frames too -- so when the enemy tilts and its
		// state changes, the new frame's collision data must also reflect
		// the accumulated damage.
		for (int f = 0; f < (int)sprite_frames.size(); ++f)
			syncChunksToParentFrame(f);
	}

	// Copy every chunk's pixel data for the given frame index back into
	// the parent enemy's sprite_frames[frame]. Safe to call with any
	// valid frame index; only that frame is touched on the parent.
	void syncChunksToParentFrame(int frame)
	{
		if (frame < 0 || frame >= (int)sprite_frames.size())
			return;
		if (sprite_frames[frame].empty())
			return;

		unsigned char* dst = sprite_frames[frame].data();
		const int cw = enemy_chunk_size_width;
		const int ch = enemy_chunk_size_height;

		for (size_t ci = 0; ci < chunks.size(); ++ci)
		{
			const enemy_chunk& c = chunks[ci];
			if (frame >= (int)c.sprite_frames.size() || c.sprite_frames[frame].empty())
				continue;

			const unsigned char* src = c.sprite_frames[frame].data();

			// The chunk covers [offset_x, offset_x+cw) x [offset_y,
			// offset_y+ch) in parent-local coords, clipped to the
			// parent's extent. (rebuildChunks padded edge chunks with
			// transparent black, so we must clip the copy here.)
			const int copyW = std::min(cw, width - c.offset_x);
			const int copyH = std::min(ch, height - c.offset_y);
			if (copyW <= 0 || copyH <= 0)
				continue;

			for (int y = 0; y < copyH; ++y)
			{
				const int parent_row = c.offset_y + y;
				const int parent_idx = (parent_row * width + c.offset_x) * 4;
				const int chunk_idx = (y * cw) * 4;
				std::memcpy(dst + parent_idx, src + chunk_idx, (size_t)copyW * 4);
			}
		}
	}

};


class boss_ship : public enemy_ship
{
public:


};


class foreground_tile : public sprite
{
public:
	foreground_tile(
		GLuint tileTex,
		int w,
		int h,
		float srcStartX,  // Position in original image coordinates
		float srcStartY,
		const vector<unsigned char>& src_raw_data)
	{
		tex = tileTex;
		width = w;
		height = h;
		x = srcStartX;  // Position in original image coordinates
		y = srcStartY;

		//raw_data = src_raw_data;
		to_present_data = src_raw_data;
	}
};

class background_tile : public sprite
{
public:

};


class bullet : public sprite
{
public:

	float birth_time = 0;
	float death_time = -1;

	float density_add = 0.00008f;
	float velocity_add = 0.00008f;

	virtual void integrate(float dt)
	{


	}
};

class sine_bullet : public bullet
{
public:

	sine_bullet(bullet& b) : bullet(b) {}

	sine_bullet(void)
	{

	}

	float sinusoidal_frequency;
	float sinusoidal_amplitude;
	bool sinusoidal_shift;

	void integrate(float dt)
	{
		old_x = x;
		old_y = y;

		// Store the original direction vector
		float dirX = vel_x * dt;
		float dirY = vel_y * dt;

		// Normalize the direction vector
		float dirLength = sqrt(dirX * dirX + dirY * dirY);
		if (dirLength > 0) {
			dirX /= dirLength;
			dirY /= dirLength;
		}

		// Calculate the perpendicular direction vector (rotate 90 degrees)
		float perpX = -dirY;
		float perpY = dirX;

		// Calculate time-based sinusoidal amplitude
		// Use the birth_time to ensure continuous motion
		float timeSinceCreation = GLOBAL_TIME - this->birth_time;
		float frequency = sinusoidal_frequency; // Controls how many waves appear
		float amplitude = sinusoidal_amplitude * dt; // Controls wave height

		float sinValue = 0;

		if (sinusoidal_shift)
			sinValue = -sin(timeSinceCreation * frequency);
		else
			sinValue = sin(timeSinceCreation * frequency);

		// Move forward along original path
		float forwardSpeed = dirLength; // Original velocity magnitude
		x += dirX * forwardSpeed;
		y += dirY * forwardSpeed;

		// Add sinusoidal motion perpendicular to the path
		x += perpX * sinValue * amplitude * dt;
		y += perpY * sinValue * amplitude * dt;
	}
};










class straight_bullet : public bullet
{
public:

	straight_bullet(bullet& b) : bullet(b) {}

	straight_bullet(void)
	{

	}

	void integrate(float dt)
	{
		//const float inv_aspect = SIM_HEIGHT / float(SIM_WIDTH);

		old_x = x;
		old_y = y;

		// Store the original direction vector
		float dirX = vel_x * dt;
		float dirY = vel_y * dt;

		// Normalize the direction vector
		float dirLength = sqrt(dirX * dirX + dirY * dirY);
		if (dirLength > 0) {
			dirX /= dirLength;
			dirY /= dirLength;
		}

		// Move forward along original path
		float forwardSpeed = dirLength; // Original velocity magnitude
		x += dirX * forwardSpeed;
		y += dirY * forwardSpeed;
	}

};







inline float pixelToNormX(float px) { return px / (float)SIM_WIDTH; }
inline float pixelToNormY(float py) { return 1.0f - py / (float)SIM_HEIGHT; }  // Flip Y

// Convert pixel velocity to normalized velocity  
inline float velPixelToNormX(float vx) { return vx / (float)SIM_WIDTH; }
inline float velPixelToNormY(float vy) { return -vy / (float)SIM_HEIGHT; }

vector<unique_ptr<bullet>> ally_bullets;
vector<unique_ptr<bullet>> enemy_bullets;

// Power-up templates (one per type). Only their tex/width/height are used as a
// blueprint when we spawn a live power_up into the world.
sprite power_up_template_sinusoidal;
sprite power_up_template_x3;
sprite power_up_template_x5;

// All power-ups currently alive in the world. Spawned when an enemy carrying
// power-ups dies; integrated, drawn, and culled each frame.
vector<unique_ptr<power_up>> power_ups_alive;



vector<unique_ptr<enemy_ship>> enemy_ships;



// These global objects do not need to generate new tex(es)
friendly_ship protagonist;
background_tile background;
background_tile background_lit;

bullet bullet_template;
vector<enemy_ship> enemy_templates;  // Automatically populated from media/enemy_<N>_<frame>.png files
boss_ship boss_template;


sprite game_over_banner;


const int foreground_chunk_size_width = 360;
const int foreground_chunk_size_height = 108;


vector<foreground_tile> foreground_chunked;
vector<foreground_tile> foreground_lit_chunked;



// OpenGL objects
GLuint velocityFBO[2], pressureFBO[2], densityFBO[2];
GLuint velocityTex[2], pressureTex[2], densityTex[2];
GLuint divergenceFBO, divergenceTex;
GLuint vorticityFBO_obj, vorticityTex;
GLuint obstacleFBO, obstacleTex;
GLuint tempFBO, tempTex;

int currentVelocity = 0;
int currentPressure = 0;
int currentDensity = 0;

// Shader programs
GLuint advectProgram;
GLuint divergenceProgram;
GLuint jacobiProgram;
GLuint gradientSubtractProgram;
GLuint addSourceProgram;
GLuint boundaryProgram;
GLuint displayProgram;
GLuint vorticityProgram;
GLuint vorticityForceProgram;
GLuint obstacleProgram;
GLuint obstacleStampProgram;
GLuint copyProgram;
GLuint spriteProgram;
GLuint turbulenceForceProgram;
GLuint healthBarProgram;

// ============== GLOW SHADER ADDITIONS ==============
// Glow effect shader programs
GLuint brightnessProgram;
GLuint blurProgram;
GLuint compositeProgram;

// Glow effect FBOs and textures
GLuint sceneFBO, sceneTex;           // Full scene render target
GLuint brightFBO, brightTex;         // Bright pixels extraction
GLuint blurFBO[2], blurTex[2];       // Ping-pong blur buffers

// Glow parameters (can be adjusted at runtime)
float glowThreshold = 0.5f;          // Brightness threshold for glow
float glowIntensity = 2.0f;          // Glow strength multiplier
int glowBlurPasses = 3;              // Number of blur iterations (more = softer glow)
bool glowEnabled = true;             // Toggle glow on/off
// ============== END GLOW ADDITIONS ==============

// ============== CHROMATIC ABERRATION ADDITIONS ==============
// Chromatic aberration shader program
GLuint chromaticAberrationProgram;

// Chromatic aberration parameters
float aberrationIntensity = 0.015f;      // RGB channel separation amount
float aberrationDuration = 1.0f;         // How long effect lasts after damage (seconds)
float vignetteStrength = 0.5f;           // Vignette darkness intensity
float vignetteRadius = 0.6f;             // Vignette inner radius (0-1)
bool chromaticAberrationEnabled = true;  // Toggle effect on/off

// Damage tracking for chromatic aberration
float lastDamageTime = -1;         // When protagonist last took damage
// ============== END CHROMATIC ABERRATION ADDITIONS ==============











bool detectSpriteOverlap(const pre_sprite& sprA, const pre_sprite& sprB, unsigned char alphaThreshold = 1)
{
	// First, do a fast bounding box test
	// sprA's bounding box
	float aLeft = sprA.x;
	float aRight = sprA.x + sprA.width;
	float aTop = sprA.y;
	float aBottom = sprA.y + sprA.height;

	// sprB's bounding box
	float bLeft = sprB.x;
	float bRight = sprB.x + sprB.width;
	float bTop = sprB.y;
	float bBottom = sprB.y + sprB.height;

	// Check for bounding box overlap
	if (aRight <= bLeft || bRight <= aLeft || aBottom <= bTop || bBottom <= aTop)
	{
		return false; // No bounding box overlap, no collision possible
	}

	// Calculate the overlapping region in screen coordinates
	int overlapLeft = static_cast<int>(std::max(aLeft, bLeft));
	int overlapRight = static_cast<int>(std::min(aRight, bRight));
	int overlapTop = static_cast<int>(std::max(aTop, bTop));
	int overlapBottom = static_cast<int>(std::min(aBottom, bBottom));

	// Get the pixel data pointers
	// Use the first available to_present_data pointer (index 0)
	// For tri_sprite, this will be one of the state textures
	if (sprA.to_present_data_pointers.empty() || sprA.to_present_data_pointers[0] == nullptr)
		return false;
	if (sprB.to_present_data_pointers.empty() || sprB.to_present_data_pointers[0] == nullptr)
		return false;

	const unsigned char* dataA = sprA.to_present_data_pointers[0];
	const unsigned char* dataB = sprB.to_present_data_pointers[0];

	// Iterate over the overlapping region
	for (int screenY = overlapTop; screenY < overlapBottom; screenY++)
	{
		for (int screenX = overlapLeft; screenX < overlapRight; screenX++)
		{
			// Convert screen coordinates to local sprite coordinates for sprite A
			int localAX = screenX - static_cast<int>(sprA.x);
			int localAY = screenY - static_cast<int>(sprA.y);

			// Convert screen coordinates to local sprite coordinates for sprite B
			int localBX = screenX - static_cast<int>(sprB.x);
			int localBY = screenY - static_cast<int>(sprB.y);

			// Bounds check (should be within bounds due to overlap calculation, but be safe)
			if (localAX < 0 || localAX >= sprA.width || localAY < 0 || localAY >= sprA.height)
				continue;
			if (localBX < 0 || localBX >= sprB.width || localBY < 0 || localBY >= sprB.height)
				continue;

			// Calculate pixel indices (RGBA format, 4 bytes per pixel)
			// Data is stored row by row from top to bottom
			size_t indexA = (static_cast<size_t>(localAY) * sprA.width + localAX) * 4;
			size_t indexB = (static_cast<size_t>(localBY) * sprB.width + localBX) * 4;

			// Get alpha values (alpha is at offset +3 in RGBA)
			unsigned char alphaA = dataA[indexA + 3];
			unsigned char alphaB = dataB[indexB + 3];

			// If both pixels are non-transparent, we have a collision
			if (alphaA >= alphaThreshold && alphaB >= alphaThreshold)
			{
				return true;
			}
		}
	}

	return false;
}



/**
 * Get the active pixel data for a tri_sprite based on its current state.
 * Helper function used by the collision detection functions.
 */
const unsigned char* getTriSpriteActiveData(const tri_sprite& spr)
{
	if (spr.sprite_frames.empty())
		return nullptr;

	int idx = spr.state;
	if (idx < 0) idx = 0;
	if (idx >= (int)spr.sprite_frames.size()) idx = (int)spr.sprite_frames.size() - 1;

	if (!spr.sprite_frames[idx].empty())
		return spr.sprite_frames[idx].data();

	// Fallback to rest frame
	int rest = (int)spr.sprite_frames.size() / 2;
	if (!spr.sprite_frames[rest].empty())
		return spr.sprite_frames[rest].data();

	return nullptr;
}

bool detectTriSpriteToSpriteOverlap(
	const tri_sprite& triSpr,
	const sprite& spr,
	unsigned char alphaThreshold = 1)
{
	// Fast bounding box test
	float aLeft = triSpr.x;
	float aRight = triSpr.x + triSpr.width;
	float aTop = triSpr.y;
	float aBottom = triSpr.y + triSpr.height;

	float bLeft = spr.x;
	float bRight = spr.x + spr.width;
	float bTop = spr.y;
	float bBottom = spr.y + spr.height;

	if (aRight <= bLeft || bRight <= aLeft || aBottom <= bTop || bBottom <= aTop)
	{
		return false;
	}

	int overlapLeft = static_cast<int>(std::max(aLeft, bLeft));
	int overlapRight = static_cast<int>(std::min(aRight, bRight));
	int overlapTop = static_cast<int>(std::max(aTop, bTop));
	int overlapBottom = static_cast<int>(std::min(aBottom, bBottom));

	// Get tri_sprite data based on current state
	const unsigned char* dataA = getTriSpriteActiveData(triSpr);

	// Get sprite data
	const unsigned char* dataB = spr.to_present_data.empty() ? nullptr : spr.to_present_data.data();

	if (dataA == nullptr || dataB == nullptr)
		return false;

	for (int screenY = overlapTop; screenY < overlapBottom; screenY++)
	{
		for (int screenX = overlapLeft; screenX < overlapRight; screenX++)
		{
			int localAX = screenX - static_cast<int>(triSpr.x);
			int localAY = screenY - static_cast<int>(triSpr.y);
			int localBX = screenX - static_cast<int>(spr.x);
			int localBY = screenY - static_cast<int>(spr.y);

			if (localAX < 0 || localAX >= triSpr.width || localAY < 0 || localAY >= triSpr.height)
				continue;
			if (localBX < 0 || localBX >= spr.width || localBY < 0 || localBY >= spr.height)
				continue;

			size_t indexA = (static_cast<size_t>(localAY) * triSpr.width + localAX) * 4;
			size_t indexB = (static_cast<size_t>(localBY) * spr.width + localBX) * 4;

			unsigned char alphaA = dataA[indexA + 3];
			unsigned char alphaB = dataB[indexB + 3];

			if (alphaA >= alphaThreshold && alphaB >= alphaThreshold)
			{
				return true;
			}
		}
	}

	return false;
}


bool detectTriSpriteOverlap(const tri_sprite& sprA, const tri_sprite& sprB, unsigned char alphaThreshold = 1)
{
	// Fast bounding box test
	float aLeft = sprA.x;
	float aRight = sprA.x + sprA.width;
	float aTop = sprA.y;
	float aBottom = sprA.y + sprA.height;

	float bLeft = sprB.x;
	float bRight = sprB.x + sprB.width;
	float bTop = sprB.y;
	float bBottom = sprB.y + sprB.height;

	if (aRight <= bLeft || bRight <= aLeft || aBottom <= bTop || bBottom <= aTop)
	{
		return false;
	}

	int overlapLeft = static_cast<int>(std::max(aLeft, bLeft));
	int overlapRight = static_cast<int>(std::min(aRight, bRight));
	int overlapTop = static_cast<int>(std::max(aTop, bTop));
	int overlapBottom = static_cast<int>(std::min(aBottom, bBottom));

	// Get the correct data pointer based on current state
	const unsigned char* dataA = getTriSpriteActiveData(sprA);
	const unsigned char* dataB = getTriSpriteActiveData(sprB);

	if (dataA == nullptr || dataB == nullptr)
		return false;

	for (int screenY = overlapTop; screenY < overlapBottom; screenY++)
	{
		for (int screenX = overlapLeft; screenX < overlapRight; screenX++)
		{
			int localAX = screenX - static_cast<int>(sprA.x);
			int localAY = screenY - static_cast<int>(sprA.y);
			int localBX = screenX - static_cast<int>(sprB.x);
			int localBY = screenY - static_cast<int>(sprB.y);

			if (localAX < 0 || localAX >= sprA.width || localAY < 0 || localAY >= sprA.height)
				continue;
			if (localBX < 0 || localBX >= sprB.width || localBY < 0 || localBY >= sprB.height)
				continue;

			size_t indexA = (static_cast<size_t>(localAY) * sprA.width + localAX) * 4;
			size_t indexB = (static_cast<size_t>(localBY) * sprB.width + localBX) * 4;

			unsigned char alphaA = dataA[indexA + 3];
			unsigned char alphaB = dataB[indexB + 3];

			if (alphaA >= alphaThreshold && alphaB >= alphaThreshold)
			{
				return true;
			}
		}
	}

	return false;
}









GLuint collisionTex = 0;      // Now RG32F -> red density in R, green in G
GLuint collisionFBO = 0;
GLuint collisionProgram = 0;
std::vector<glm::vec4> collisionPoints;

const char* collisionFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec2 fragColor;        // .r = red density, .g = green density

uniform sampler2D density;      // RG = red / green density
uniform sampler2D obstacles;
uniform vec2 texelSize;

void main()
{
    float obstC = texture(obstacles, texCoord).r;

    // Inside solid obstacle -> no collision
    if (obstC > 0.5) {
        fragColor = vec2(0.0);
        return;
    }

    // Sample 4 neighbors
    float oL = texture(obstacles, texCoord - vec2(texelSize.x, 0.0)).r;
    float oR = texture(obstacles, texCoord + vec2(texelSize.x, 0.0)).r;
    float oB = texture(obstacles, texCoord - vec2(0.0, texelSize.y)).r;
    float oT = texture(obstacles, texCoord + vec2(0.0, texelSize.y)).r;

    bool isEdge = (oL > 0.5) || (oR > 0.5) || (oB > 0.5) || (oT > 0.5);

    if (!isEdge) {
        fragColor = vec2(0.0);
        return;
    }

    // This pixel is fluid touching an obstacle edge -> report density
    vec2 dens = texture(density, texCoord).rg;  // <-- using texture() correctly in shader
    fragColor = dens;  // red in .r, green in .g
}
)";





// Utility functions
void checkGLError(const char* operation) {
	GLenum error = glGetError();
	if (error != GL_NO_ERROR) {
		std::cerr << "OpenGL error after " << operation << ": " << error << std::endl;
	}
}

GLuint compileShader(GLenum type, const char* source) {
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
		std::cerr << "Source:\n" << source << std::endl;
	}

	return shader;
}



const char* textVertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec4 aColor;

uniform mat4 projection;
uniform mat4 model;

out vec2 TexCoord;
out vec4 Color;

void main() {
    gl_Position = projection * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    Color = aColor;
}
)";

const char* textFragmentShaderSource = R"(
#version 330 core
in vec2 TexCoord;
in vec4 Color;

uniform sampler2D fontTexture;
uniform bool useColor;

out vec4 FragColor;

void main() {
    vec4 texColor = texture(fontTexture, TexCoord);
    
    // Handle different font texture formats
    if (useColor) {
        // For colored font atlas, just blend with the vertex color
        FragColor = texColor * Color;
    } else {
        // For grayscale/alpha font atlas, use the alpha/red channel
        // with the vertex color
        float alpha = texColor.r; // or texColor.a depending on your font texture
        FragColor = vec4(Color.rgb, Color.a * alpha);
    }
}
)";






struct FontAtlas {
	GLuint textureID;
	int charWidth;      // Width of each character (16)
	int charHeight;     // Height of each character (16)
	int atlasWidth;     // Atlas width (256)
	int atlasHeight;    // Atlas height (256)
	int charsPerRow;    // Characters per row in the atlas (16)
};


GLuint loadFontTexture(const char* filename) {
	GLuint textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Load image using stb_image (you're already using this)
	int width, height, channels;
	stbi_set_flip_vertically_on_load(false);


	unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);

	if (!data) {
		std::cerr << "Failed to load font texture: " << filename << std::endl;
		std::cerr << "STB Image error: " << stbi_failure_reason() << std::endl;
		return 0;
	}

	// Determine format based on channels
	GLenum format;
	switch (channels) {
	case 1: format = GL_RED; break;
	case 3: format = GL_RGB; break;
	case 4: format = GL_RGBA; break;
	default:
		format = GL_RGB;
		std::cerr << "Unsupported number of channels: " << channels << std::endl;
	}

	// Load texture data to GPU
	glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

	// Free image data
	stbi_image_free(data);

	return textureID;
}

FontAtlas initFontAtlas(const char* filename) {
	FontAtlas atlas;
	atlas.textureID = loadFontTexture(filename);
	atlas.charWidth = 64;
	atlas.charHeight = 64;
	atlas.atlasWidth = 1024;
	atlas.atlasHeight = 1024;
	atlas.charsPerRow = atlas.atlasWidth / atlas.charWidth; // 16

	return atlas;
}







bool g_editorMode = false;
float g_editorFgScrollSpeed = 600.0f; // pixels per second when arrow key held

int  g_selectedEnemy = 0;
int  g_selectedPoint = -1;
bool g_draggingPoint = false;
int  g_selectedSpeedKnot = -1;   // index into path_speeds, -1 = none
int  g_selectedCannon = -1;      // index into selected enemy's cannons, -1 = none
int  g_selectedPowerUp = -1;     // index into selected enemy's power_ups, -1 = none
int  g_spawnTemplateIdx = 0;
bool g_dragUndoPushed = false;   // true once editorPushUndo has been called for the current drag

// Clipboard for copy/paste of path data (Ctrl+C / Ctrl+V)
std::vector<glm::vec2> g_clipboard_path_points;
std::vector<float>     g_clipboard_path_speeds;
bool                   g_clipboard_has_data = false;




class TextRenderer {
private:
	FontAtlas atlas;
	GLuint VAO, VBO, EBO;
	GLuint shaderProgram;
	glm::mat4 projection;

	struct Vertex {
		glm::vec3 position;
		glm::vec2 texCoord;
		glm::vec4 color;
	};

public:

	std::unordered_map<char, int> charWidths; // Map to store actual widths

	// Add this method to calculate character widths
	void calculateCharacterWidths() {
		// Read back the font texture data from GPU
		int dataSize = atlas.atlasWidth * atlas.atlasHeight * 4; // RGBA format
		std::vector<unsigned char> textureData(dataSize);

		glBindTexture(GL_TEXTURE_2D, atlas.textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData.data());





		// Analyze each character
		for (unsigned short c_ = 0; c_ < 256; c_++)
		{
			unsigned char c = static_cast<unsigned char>(c_);

			// ASCII range
			int atlasX = (c % atlas.charsPerRow) * atlas.charWidth;
			int atlasY = (c / atlas.charsPerRow) * atlas.charHeight;

			// Find leftmost non-empty column
			int leftEdge = atlas.charWidth - 1; // Start from rightmost position

			// Find rightmost non-empty column
			int rightEdge = 0; // Start from leftmost position

			// Scan all columns for this character
			for (int x = 0; x < atlas.charWidth; x++) {
				bool columnHasPixels = false;

				// Check if any pixel in this column is non-transparent
				for (int y = 0; y < atlas.charHeight; y++) {
					int pixelIndex = ((atlasY + y) * atlas.atlasWidth + (atlasX + x)) * 4;
					if (pixelIndex >= 0 && pixelIndex < dataSize - 3) {
						// Check alpha value (using red channel for grayscale font)
						if (textureData[pixelIndex] > 20) { // Non-transparent threshold
							columnHasPixels = true;
							break;
						}
					}
				}

				if (columnHasPixels) {
					// Update left edge (minimum value)
					leftEdge = std::min(leftEdge, x);
					// Update right edge (maximum value)
					rightEdge = std::max(rightEdge, x);
				}
			}

			// If no pixels were found (space or empty character)
			if (rightEdge < leftEdge) {
				// Default width for space character
				if (c == ' ') {
					charWidths[c] = atlas.charWidth / 3; // Make space 1/3 of cell width
				}
				else {
					charWidths[c] = atlas.charWidth / 4; // Default minimum width
				}
			}
			else {
				// Calculate width based on the actual character bounds
				int actualWidth = (rightEdge - leftEdge) + 1;

				// Add some padding
				int paddedWidth = actualWidth + 4; // 2 pixels on each side

				// Store this character's width (minimum width of 1/4 of the cell)
				charWidths[c] = std::max(paddedWidth, atlas.charWidth / 4);
			}
		}
	}


	TextRenderer(const char* fontAtlasFile, int windowWidth, int windowHeight) {
		// Initialize font atlas
		atlas = initFontAtlas(fontAtlasFile);

		// Create shader program
		GLuint vertexShader = compileShader(GL_VERTEX_SHADER, textVertexShaderSource);
		GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, textFragmentShaderSource);

		shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);

		// Check for linking errors
		GLint success;
		glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
		if (!success) {
			GLchar infoLog[512];
			glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
			std::cerr << "Shader program linking error: " << infoLog << std::endl;
		}

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		// Create VAO, VBO, EBO for text rendering
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		// Set up vertex attributes
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3)));
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3) + sizeof(glm::vec2)));
		glEnableVertexAttribArray(2);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

		// Set up projection matrix
		setProjection(windowWidth, windowHeight);

		calculateCharacterWidths();
	}

	~TextRenderer() {
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &EBO);
		glDeleteVertexArrays(1, &VAO);
		glDeleteProgram(shaderProgram);
		glDeleteTextures(1, &atlas.textureID);
	}

	void setProjection(int windowWidth, int windowHeight) {
		projection = glm::ortho(0.0f, (float)windowWidth, (float)windowHeight, 0.0f, -1.0f, 1.0f);
	}














	void renderText(const std::string& text, float x, float y, float scale, glm::vec4 color, bool centered = false) {
		glUseProgram(shaderProgram);

		// Set uniforms
		GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
		glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

		GLuint fontTexLoc = glGetUniformLocation(shaderProgram, "fontTexture");
		glUniform1i(fontTexLoc, 0);

		GLuint useColorLoc = glGetUniformLocation(shaderProgram, "useColor");
		glUniform1i(useColorLoc, 0); // Set to 1 if your font atlas is colored

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, atlas.textureID);

		glBindVertexArray(VAO);

		// If text should be centered, calculate the starting position
		if (centered) {
			float textWidth = 0;
			for (char c : text) {
				// Use the calculated width for each character
				float charWidth = static_cast<float>(charWidths[c]);// charWidths.count(c) ? charWidths[c] : atlas.charWidth / 2;
				textWidth += (8 + charWidth) * scale;
			}
			x = windowWidth / 2.0f - textWidth / 2.0f;
		}

		// For each character, create a quad with appropriate texture coordinates
		std::vector<Vertex> vertices;
		std::vector<GLuint> indices;

		float xpos = x;
		float ypos = y;
		GLuint indexOffset = 0;

		for (char c : text) {
			// Get ASCII value of the character
			unsigned char charValue = static_cast<unsigned char>(c);

			// Calculate position in the atlas using ASCII value
			int atlasX = (charValue % atlas.charsPerRow) * atlas.charWidth;
			int atlasY = (charValue / atlas.charsPerRow) * atlas.charHeight;

			// Calculate texture coordinates
			float texLeft = atlasX / (float)atlas.atlasWidth;
			float texRight = (atlasX + atlas.charWidth) / (float)atlas.atlasWidth;
			float texTop = atlasY / (float)atlas.atlasHeight;
			float texBottom = (atlasY + atlas.charHeight) / (float)atlas.atlasHeight;

			// Get the character's calculated width
			float charWidth = static_cast<float>(charWidths[charValue]);// charWidths.count(charValue) ? charWidths[charValue] : atlas.charWidth / 2;

			// Calculate quad vertices
			float quadLeft = xpos;
			float quadRight = xpos + atlas.charWidth * scale; // Use full cell width for texture
			float quadTop = ypos;
			float quadBottom = ypos + atlas.charHeight * scale;

			// Add vertices
			vertices.push_back({ {quadLeft, quadTop, 0.0f}, {texLeft, texTop}, color });
			vertices.push_back({ {quadRight, quadTop, 0.0f}, {texRight, texTop}, color });
			vertices.push_back({ {quadRight, quadBottom, 0.0f}, {texRight, texBottom}, color });
			vertices.push_back({ {quadLeft, quadBottom, 0.0f}, {texLeft, texBottom}, color });

			// Add indices
			indices.push_back(indexOffset + 0);
			indices.push_back(indexOffset + 1);
			indices.push_back(indexOffset + 2);
			indices.push_back(indexOffset + 0);
			indices.push_back(indexOffset + 2);
			indices.push_back(indexOffset + 3);

			indexOffset += 4;

			// Advance cursor using the calculated width
			// add 8 pixels of padding between characters
			xpos += (8 + charWidth) * scale;
		}

		// Upload vertex and index data
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_DYNAMIC_DRAW);

		// Draw text
		glm::mat4 model = glm::mat4(1.0f);
		GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

		// Enable blending for transparent font
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);

		// Reset state
		glDisable(GL_BLEND);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
};


int editorFindTemplateIdx(const enemy_ship* e)
{
	return e->template_idx;
}

TextRenderer* textRenderer = nullptr;



static ostringstream editorPrintState(int g_selectedEnemy)
{
	ostringstream oss;

	oss << "\n========== EDITOR STATE ==========\n";
	//for (size_t i = 0; i < enemy_ships.size(); ++i)

	int i = g_selectedEnemy;
	{
		const enemy_ship& e = *enemy_ships[i];
		oss << "Enemy " << i << ":\n";
		oss << "  max_health = " << e.max_health << "\n";
		oss << "  path_animation_length = " << e.path_animation_length << "s\n";
		oss << "  Path points (" << e.path_points.size() << "):\n";
		for (size_t j = 0; j < e.path_points.size(); ++j)
		{
			const char* marker = ((int)j == g_selectedPoint) ? ">> " : "   ";
			oss << "  " << marker << "[" << j << "] x=" << e.path_points[j].x / SIM_WIDTH
				<< " y=" << e.path_points[j].y / SIM_HEIGHT << " (norm)\n";
		}

		oss << "  Speed knots (" << e.path_speeds.size() << "):\n";
		for (size_t j = 0; j < e.path_speeds.size(); ++j)
		{
			const char* marker = ((int)j == g_selectedSpeedKnot) ? ">> " : "   ";
			oss << "  " << marker << "[" << j << "] " << e.path_speeds[j] << "\n";
		}

		oss << "  Cannons (" << e.cannons.size() << "):\n";
		for (size_t j = 0; j < e.cannons.size(); ++j)
		{
			const char* tname = "LEFT";
			if (e.cannons[j].cannon_type == CANNON_TYPE_UP_DOWN)  tname = "UP_DOWN";
			if (e.cannons[j].cannon_type == CANNON_TYPE_TRACKING) tname = "TRACKING";
			if (e.cannons[j].cannon_type == CANNON_TYPE_CIRCULAR) tname = "CIRCULAR";
			// Selected cannon gets a ">>" marker so renderEditorOverlay can
			// highlight that line in a different color.
			const char* marker = ((int)j == g_selectedCannon) ? ">> " : "   ";
			oss << "  " << marker << "[" << j << "] type=" << tname
				<< " x=" << e.cannons[j].x / std::max(1, e.width - 1) << " (norm)"
				<< " y=" << e.cannons[j].y / std::max(1, e.height - 1) << " (norm)"
				<< " interval=" << e.cannons[j].min_bullet_interval << "s\n";
		}

		oss << "  Power-ups (" << e.power_ups.size() << "):\n";
		for (size_t j = 0; j < e.power_ups.size(); ++j)
		{
			const char* pname = "SINUSOIDAL";
			if (e.power_ups[j] == POWER_UP_TYPE_X3) pname = "X3";
			if (e.power_ups[j] == POWER_UP_TYPE_X5) pname = "X5";
			// Selected power-up gets a ">>" marker.
			const char* marker = ((int)j == g_selectedPowerUp) ? ">> " : "   ";
			oss << "  " << marker << "[" << j << "] type=" << pname << "\n";
		}
	}
	oss << "==================================\n\n";

	//cout << oss.str() << endl;

	return oss;
}



void displayFPS()
{
	static int frame_count = 0;
	static float lastTime = 0.0f;
	static float fps = 0.0f;

	frame_count++;

	float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	float deltaTime = currentTime - lastTime;

	if (deltaTime >= 1.0f)
	{
		fps = frame_count / deltaTime;
		frame_count = 0;
		lastTime = currentTime;
	}

	std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));

	if (textRenderer)
		textRenderer->renderText(fpsText, 0.0, 10, 0.5f, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), true);
}









// Shader source code embedded as strings
const char* vertexShaderSource = R"(
#version 400 core
layout(location = 0) in vec2 position;
out vec2 texCoord;
void main() {
    texCoord = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
)";










const char* advectFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D quantity;
uniform sampler2D obstacles;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;

void main() {
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = dissipation * texture(quantity, texCoord);
		//fragColor = vec4(0.0);
        return;
    }
    
    vec2 vel = texture(velocity, texCoord).xy;
    vec2 pos = texCoord - vel * texelSize * dt;
    
    // Clamp to valid texture coordinates
    pos = clamp(pos, texelSize * 0.5, 1.0 - texelSize * 0.5);
    
    fragColor = dissipation * texture(quantity, pos);
}
)";

const char* divergenceFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D obstacles;
uniform vec2 texelSize;

void main() {
    float oC = texture(obstacles, texCoord).x;

    if (oC > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    vec2 vL = texture(velocity, texCoord - vec2(texelSize.x, 0.0)).xy;
    vec2 vR = texture(velocity, texCoord + vec2(texelSize.x, 0.0)).xy;
    vec2 vB = texture(velocity, texCoord - vec2(0.0, texelSize.y)).xy;
    vec2 vT = texture(velocity, texCoord + vec2(0.0, texelSize.y)).xy;
    
    float oL = texture(obstacles, texCoord - vec2(texelSize.x, 0.0)).x;
    float oR = texture(obstacles, texCoord + vec2(texelSize.x, 0.0)).x;
    float oB = texture(obstacles, texCoord - vec2(0.0, texelSize.y)).x;
    float oT = texture(obstacles, texCoord + vec2(0.0, texelSize.y)).x;
    
    // Use center velocity for obstacle boundaries (no-slip)
    vec2 vC = texture(velocity, texCoord).xy;
    if (oL > 0.5) vL = -vC;
    if (oR > 0.5) vR = -vC;
    if (oB > 0.5) vB = -vC;
    if (oT > 0.5) vT = -vC;
    
    float divergence = 0.5 * ((vR.x - vL.x) + (vT.y - vB.y));
    fragColor = vec4(divergence, 0.0, 0.0, 1.0);
}
)";

const char* jacobiFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D pressure;
uniform sampler2D divergence;
uniform sampler2D obstacles;
uniform vec2 texelSize;
uniform float alpha;
uniform float rBeta;

void main() {
    float oC = texture(obstacles, texCoord).x;
    if (oC > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    float pL = texture(pressure, texCoord - vec2(texelSize.x, 0.0)).x;
    float pR = texture(pressure, texCoord + vec2(texelSize.x, 0.0)).x;
    float pB = texture(pressure, texCoord - vec2(0.0, texelSize.y)).x;
    float pT = texture(pressure, texCoord + vec2(0.0, texelSize.y)).x;
    float pC = texture(pressure, texCoord).x;
    
    float oL = texture(obstacles, texCoord - vec2(texelSize.x, 0.0)).x;
    float oR = texture(obstacles, texCoord + vec2(texelSize.x, 0.0)).x;
    float oB = texture(obstacles, texCoord - vec2(0.0, texelSize.y)).x;
    float oT = texture(obstacles, texCoord + vec2(0.0, texelSize.y)).x;
    
    // Use center pressure for obstacle boundaries
    if (oL > 0.5) pL = pC;
    if (oR > 0.5) pR = pC;
    if (oB > 0.5) pB = pC;
    if (oT > 0.5) pT = pC;
    
    float bC = texture(divergence, texCoord).x;
    fragColor = vec4((pL + pR + pB + pT + alpha * bC) * rBeta, 0.0, 0.0, 1.0);
}
)";

const char* gradientSubtractFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D pressure;
uniform sampler2D velocity;
uniform sampler2D obstacles;
uniform vec2 texelSize;

void main() {
    float oC = texture(obstacles, texCoord).x;
    if (oC > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    float pL = texture(pressure, texCoord - vec2(texelSize.x, 0.0)).x;
    float pR = texture(pressure, texCoord + vec2(texelSize.x, 0.0)).x;
    float pB = texture(pressure, texCoord - vec2(0.0, texelSize.y)).x;
    float pT = texture(pressure, texCoord + vec2(0.0, texelSize.y)).x;
    
    float oL = texture(obstacles, texCoord - vec2(texelSize.x, 0.0)).x;
    float oR = texture(obstacles, texCoord + vec2(texelSize.x, 0.0)).x;
    float oB = texture(obstacles, texCoord - vec2(0.0, texelSize.y)).x;
    float oT = texture(obstacles, texCoord + vec2(0.0, texelSize.y)).x;
    
    float pC = texture(pressure, texCoord).x;
    
    // Handle obstacle boundaries
    vec2 obstVel = vec2(0.0);
    vec2 vMask = vec2(1.0);
    
    if (oL > 0.5) { pL = pC; vMask.x = 0.0; }
    if (oR > 0.5) { pR = pC; vMask.x = 0.0; }
    if (oB > 0.5) { pB = pC; vMask.y = 0.0; }
    if (oT > 0.5) { pT = pC; vMask.y = 0.0; }
    
    vec2 vel = texture(velocity, texCoord).xy;
    vec2 grad = vec2(pR - pL, pT - pB) * 0.5;
    vec2 newVel = vel - grad;
    
    fragColor = vec4(vMask * newVel + obstVel, 0.0, 1.0);
}
)";

const char* addSourceFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D field;
uniform sampler2D obstacles;
uniform vec2 point;
uniform vec3 value;
uniform float radius;
uniform vec2 texelSize;
uniform vec2 aspectRatio;

void main() {
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = texture(field, texCoord);
        return;
    }
    
    vec2 diff = (texCoord - point) * aspectRatio;
    float dist = length(diff);
    float factor = exp(-dist * dist / radius);
    
    vec4 current = texture(field, texCoord);
    fragColor = current + factor * vec4(value, 0.0);
}
)";

const char* vorticityFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D obstacles;
uniform vec2 texelSize;

void main() {
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    float vL = texture(velocity, texCoord - vec2(texelSize.x, 0.0)).y;
    float vR = texture(velocity, texCoord + vec2(texelSize.x, 0.0)).y;
    float vB = texture(velocity, texCoord - vec2(0.0, texelSize.y)).x;
    float vT = texture(velocity, texCoord + vec2(0.0, texelSize.y)).x;
    
    float vorticity = ((vR - vL) - (vT - vB)) * 0.5;
    fragColor = vec4(vorticity, 0.0, 0.0, 1.0);
}
)";

const char* vorticityForceFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D vorticity;
uniform sampler2D obstacles;
uniform vec2 texelSize;
uniform float dt;
uniform float scale;

void main() {
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    float vL = texture(vorticity, texCoord - vec2(texelSize.x, 0.0)).x;
    float vR = texture(vorticity, texCoord + vec2(texelSize.x, 0.0)).x;
    float vB = texture(vorticity, texCoord - vec2(0.0, texelSize.y)).x;
    float vT = texture(vorticity, texCoord + vec2(0.0, texelSize.y)).x;
    float vC = texture(vorticity, texCoord).x;
    
    vec2 force = vec2(abs(vT) - abs(vB), abs(vR) - abs(vL)) * 0.5;
    float len = length(force);
    if (len > 0.0001) {
        force = force / len;
    }
    force *= scale * vC * vec2(1.0, -1.0);
    
    vec2 vel = texture(velocity, texCoord).xy;
    fragColor = vec4(vel + force * dt, 0.0, 1.0);
}
)";

const char* obstacleFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D obstacles;
uniform vec2 point;
uniform float radius;
uniform vec2 texelSize;
uniform vec2 aspectRatio;
uniform float addOrRemove;

void main() {
    vec2 diff = (texCoord - point) * aspectRatio;
    float dist = length(diff);
    
    float current = texture(obstacles, texCoord).x;
    
    if (dist < radius) {
        fragColor = vec4(addOrRemove, 0.0, 0.0, 1.0);
    } else {
        fragColor = vec4(current, 0.0, 0.0, 1.0);
    }
}
)";

// Obstacle stamp shader - uses a sprite texture as a stamp
const char* obstacleStampFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D obstacles;      // Current obstacle field
uniform sampler2D stampTexture;   // Sprite texture to stamp
uniform vec2 stampCenter;         // Center position in normalized coords [0,1]
uniform vec2 stampHalfSize;       // Half-size of stamp in normalized coords (already aspect-corrected)
uniform float threshold;          // Alpha/intensity threshold for stamp (default 0.5)
uniform float addOrRemove;        // 1.0 to add obstacles, 0.0 to remove
uniform int useAlpha;             // 1 = use alpha channel, 0 = use red/luminance

void main() {
    float current = texture(obstacles, texCoord).x;
    
    // Calculate position relative to stamp center in normalized texture space
    vec2 diff = texCoord - stampCenter;
    
    // Check if we're within the stamp bounds
    // stampHalfSize is pre-computed to account for aspect ratio
    vec2 relPos = diff / (stampHalfSize * 2.0) + 0.5;
    
    if (relPos.x >= 0.0 && relPos.x <= 1.0 && relPos.y >= 0.0 && relPos.y <= 1.0) {
        // Sample the stamp texture (flip Y for typical image coordinates)
        vec2 stampUV = vec2(relPos.x, 1.0 - relPos.y);
        vec4 stampSample = texture(stampTexture, stampUV);
        
        // Get stamp value based on mode
        float stampValue;
        if (useAlpha == 1) {
            stampValue = stampSample.a;
        } else {
            // Use luminance of RGB
            stampValue = dot(stampSample.rgb, vec3(0.299, 0.587, 0.114));
        }
        
        // Apply threshold
        if (stampValue >= threshold) {
            if (addOrRemove > 0.5) {
                // Add mode: set obstacle where stamp is solid
                fragColor = vec4(1.0, 0.0, 0.0, 1.0);
            } else {
                // Remove mode: clear obstacle where stamp is solid
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
        } else {
            // Below threshold, keep current value
            fragColor = vec4(current, 0.0, 0.0, 1.0);
        }
    } else {
        // Outside stamp bounds, keep current value
        fragColor = vec4(current, 0.0, 0.0, 1.0);
    }
}
)";

const char* displayFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D density;
uniform sampler2D velocity;
uniform sampler2D obstacles;
uniform sampler2D background;
uniform sampler2D background_lit;

uniform vec2 texelSize;
uniform float time;


float WIDTH = texelSize.x;
float HEIGHT = texelSize.y;
float aspect_ratio = WIDTH/HEIGHT;



vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() 
{
    float obstacle = texture(obstacles, texCoord).x;
    
    if (obstacle > 0.5) 
    {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }


    vec2 adjustedCoord = texCoord;
    
    // For non-square textures, adjust sampling to prevent stretching
    if (1.0/aspect_ratio > 1.0) {
        adjustedCoord.x = (adjustedCoord.x - 0.5) * aspect_ratio + 0.5;
    } else if (1.0/aspect_ratio < 1.0) {
        adjustedCoord.y = (adjustedCoord.y - 0.5) / aspect_ratio + 0.5;
    }

    vec2 scrolledCoord = adjustedCoord;
    scrolledCoord.x += time * 0.01;

    vec4 bgColor = texture(background, scrolledCoord);
	vec4 bgColor_lit = texture(background_lit, scrolledCoord);
	float t = (sin(time * 2.0f) + 1.0f) / 2.0f;

	bgColor = mix(bgColor, bgColor_lit, t);
  
    // Get density and colors at adjusted position
    float redIntensity = texture(density, texCoord).x;
    float blueIntensity = texture(density, texCoord).y;

    float d = redIntensity + blueIntensity;

    // Create color vectors based on intensity
    vec4 redFluidColor = vec4(redIntensity, 0.0, 0.0, redIntensity);
    vec4 blueFluidColor = vec4(0.0, 0.0, blueIntensity, blueIntensity);

    // Combine both colors
    vec4 combinedColor = redFluidColor + blueFluidColor;

    vec4 blendedBackground = bgColor;

    vec4 color1 = blendedBackground;
    vec4 color2 = vec4(0.0, 0.125, 0.25, 1.0);
    vec4 color3 = combinedColor;
    vec4 color4 = vec4(0.0, 0.0, 0.0, 1.0);

	// render white on top of black always
    if(length(blueFluidColor.b) > 0.5)
        color4 = vec4(1.0, 1.0, 1.0, 0.0);        
    else
		color4 = vec4(0.0, 0.0, 0.0, 0.0);

    // toon shading:
    if (d < 0.5) {
        fragColor = color1;
    } else if (d < 0.7) {
        fragColor = color2;
    } else if (d < 0.9) {
        fragColor = color3;
    } else {
        fragColor = color4;
    }
}
)";

const char* copyFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D source;

void main() {
    fragColor = texture(source, texCoord);
}
)";

// Sprite drawing shader - draws a texture at a specific screen position
const char* spriteVertexSource = R"(
#version 400 core
layout(location = 0) in vec2 position;
out vec2 texCoord;

uniform vec2 spritePos;      // Position in normalized device coords [-1, 1]
uniform vec2 spriteSize;     // Size in normalized device coords

void main() {
    // position is in [-1, 1] range for the quad
    // Map to sprite position and size
    vec2 pos = spritePos + (position * 0.5 + 0.5) * spriteSize;
    texCoord = position * 0.5 + 0.5;
    // Flip texCoord.y for correct image orientation (top-left origin for images)
    texCoord.y = 1.0 - texCoord.y;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

const char* spriteFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform float alpha = 1.0;

uniform sampler2D spriteTexture;
uniform int under_fire;
uniform float time;

void main() {
    vec4 color = texture(spriteTexture, texCoord);

	// Do alternating colour / white blinking when under fire
	if(under_fire == 1)
	{
		const float timeslice = 0.25;
		float m = mod(time, timeslice);
		
		if(m < timeslice/2.0)
		color.rgb = vec3((color.r + 1.0) * 0.5 , (color.g + 1.0) * 0.5, (color.b + 1.0) * 0.5);
	}

    fragColor = color;
	fragColor.a = min(fragColor.a, alpha);
}
)";

// Health bar shaders - simple colored quad rendering
const char* healthBarVertexSource = R"(
#version 400 core
layout(location = 0) in vec2 position;

uniform vec2 barPos;       // Position in normalized device coords [-1, 1]
uniform vec2 barSize;      // Size in normalized device coords

void main() {
    // position is in [-1, 1] range for the quad
    // Map to bar position and size
    vec2 pos = barPos + (position * 0.5 + 0.5) * barSize;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

const char* healthBarFragmentSource = R"(
#version 400 core
out vec4 fragColor;

uniform vec4 barColor;     // RGBA color of the bar

void main() {
    fragColor = barColor;
}
)";

const char* boundaryFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D field;
uniform vec2 texelSize;
uniform float scale;

void main() {
    vec2 coord = texCoord;
    float boundary = 0.0;
    vec2 offset = vec2(0.0);
    
    // Left boundary
    if (coord.x < texelSize.x) {
        offset.x = texelSize.x;
        boundary = 1.0;
    }
    // Right boundary
    else if (coord.x > 1.0 - texelSize.x) {
        offset.x = -texelSize.x;
        boundary = 1.0;
    }
    // Bottom boundary
    if (coord.y < texelSize.y) {
        offset.y = texelSize.y;
        boundary = 1.0;
    }
    // Top boundary
    else if (coord.y > 1.0 - texelSize.y) {
        offset.y = -texelSize.y;
        boundary = 1.0;
    }
    
    if (boundary > 0.5) {
        fragColor = scale * texture(field, coord + offset);
    } else {
        fragColor = texture(field, coord);
    }
}
)";

// ============== GLOW SHADER SOURCES ==============

// Brightness extraction shader - extracts pixels above threshold
const char* brightnessFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D scene;
uniform float threshold;

void main() {
    vec4 color = texture(scene, texCoord);
    
    // Calculate luminance using standard weights
    float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    
    // Extract bright pixels with smooth falloff
    float brightness = max(0.0, luminance - threshold);
    brightness = brightness / (brightness + 1.0); // Soft knee compression
    
    // Scale color by brightness factor
    fragColor = vec4(color.rgb * brightness * 2.0, 1.0);
}
)";

// Gaussian blur shader - single direction (horizontal or vertical)
const char* blurFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D image;
uniform vec2 direction;  // (1,0) for horizontal, (0,1) for vertical
uniform vec2 texelSize;

// 9-tap Gaussian weights (sigma ~= 2.5)
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec3 result = texture(image, texCoord).rgb * weights[0];
    
    for (int i = 1; i < 5; i++) {
        vec2 offset = direction * texelSize * float(i) * 2.0;
        result += texture(image, texCoord + offset).rgb * weights[i];
        result += texture(image, texCoord - offset).rgb * weights[i];
    }
    
    fragColor = vec4(result, 1.0);
}
)";

// Composite shader - combines original scene with glow
const char* compositeFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D scene;
uniform sampler2D glow;
uniform float intensity;

void main() {
    vec3 sceneColor = texture(scene, texCoord).rgb;
    vec3 glowColor = texture(glow, texCoord).rgb;
    
    // Additive blending with intensity control
    vec3 result = sceneColor + glowColor * intensity;
    
    fragColor = vec4(result, 1.0);
}
)";

// ============== END GLOW SHADER SOURCES ==============

// ============== CHROMATIC ABERRATION SHADER SOURCE ==============
const char* chromaticAberrationFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D scene;
uniform vec2 protagonistPos;     // Protagonist position in UV coords (0-1)
uniform float intensity;         // Aberration strength (0-1)
uniform float effectStrength;    // Overall effect strength (fades over time)
uniform float vignetteStrength;  // Vignette darkness
uniform float vignetteRadius;    // Vignette inner radius
uniform float time;              // For subtle animation
uniform float aspectRatio;  // Add this uniform

void main() {
    // Direction from protagonist to current pixel
    vec2 toCenter = texCoord - protagonistPos;
    float dist = length(toCenter);
    
    // Normalize direction (avoid division by zero)
    vec2 dir = dist > 0.001 ? normalize(toCenter) : vec2(0.0);
    
    // Calculate aberration offset based on distance from protagonist
    // Closer to protagonist = less aberration, further = more
    float aberrationAmount = intensity * effectStrength * dist;
    
    // Add subtle pulsing animation
    //float pulse = 1.0 + 0.2 * sin(time * 15.0) * effectStrength;
    //aberrationAmount *= 0.1*pulse;
    
    // Sample RGB channels with different offsets (radial aberration)
    vec2 redOffset = dir * aberrationAmount * 1.0;
    vec2 greenOffset = vec2(0.0);  // Green stays centered
    vec2 blueOffset = -dir * aberrationAmount * 1.0;
    
    float r = texture(scene, texCoord + redOffset).r;
    float g = texture(scene, texCoord + greenOffset).g;
    float b = texture(scene, texCoord + blueOffset).b;
    
    vec3 color = vec3(r, g, b);
    
    // Vignette effect centered on protagonist
//    float vignetteDist = length(toCenter);

vec2 aspectCorrected = toCenter;
aspectCorrected.x *= aspectRatio;
float vignetteDist = length(aspectCorrected);


    
    // Smooth vignette falloff
    float vignette = 1.0 - smoothstep(vignetteRadius, vignetteRadius + 0.4, vignetteDist);
    vignette = mix(1.0, vignette, vignetteStrength * effectStrength);
    
    // Apply vignette (darken edges relative to protagonist)
    color *= vignette;
    
    // Add slight red tint during damage for "hurt" effect
    float redTint = 0.3 * effectStrength;
    color.r = min(1.0, color.r + redTint);
    
    // Slight desaturation during damage
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(color, vec3(gray), 0.2 * effectStrength);
    
    fragColor = vec4(color, 1.0);
}
)";
// ============== END CHROMATIC ABERRATION SHADER SOURCE ==============





const char* turbulenceForceFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D obstacles;
uniform vec2 texelSize;
uniform float time;
uniform float amplitude;     // Noise amplitude
uniform float frequency;     // Noise frequency
uniform float scale;         // Overall turbulence strength

// 2D Simplex Noise function (better performance than classic Perlin)
vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                    + i.x + vec3(0.0, i1.x, 1.0));
    
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy),
                           dot(x12.zw, x12.zw)), 0.0);
    m = m*m;
    m = m*m;
    
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Generate curl noise from scalar noise field
vec2 curlNoise(vec2 uv) {
    float eps = 0.001;  // Smaller epsilon for better derivatives
    
    // Sample noise at four offset positions to compute curl
    float n0 = snoise(uv);
    float n1 = snoise(uv + vec2(eps, 0.0));
    float n2 = snoise(uv + vec2(0.0, eps));
    
    // Compute gradient using central differences
    float dx = (n1 - n0) / eps;
    float dy = (n2 - n0) / eps;
    
    // Curl in 2D
    return vec2(dy, -dx);
}

void main() {
    if(texture(obstacles, texCoord).r > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    // Get current velocity
    vec2 vel = texture(velocity, texCoord).xy;
    
    // Calculate turbulence domain with time animation
    vec2 uv = texCoord * frequency;
    
    // Add time-based scrolling for animated turbulence
    uv += vec2(time * 0.1, time * 0.15);
    
    // Generate turbulence force using curl noise
    vec2 turbulence = curlNoise(uv) * amplitude;
    
    // Add multiple octaves for richer turbulence (optional)
    vec2 turbulence2 = curlNoise(uv * 1.8 + vec2(time * 0.05)) * amplitude * 0.5;
    vec2 turbulence3 = curlNoise(uv * 3.2 - vec2(time * 0.08)) * amplitude * 0.25;
    
    // Combine octaves
    turbulence += turbulence2 + turbulence3;
    
    // Apply turbulence to velocity (scaled by dt is handled in main code)
    vel += turbulence * scale;
    
    fragColor = vec4(vel, 0.0, 1.0);
}
)";





GLuint createProgram(const char* vertexSource, const char* fragmentSource) {
	GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
	GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

	GLuint program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	GLint success;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		std::cerr << "Program linking failed:\n" << infoLog << std::endl;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return program;
}

void createTexture(GLuint& tex, int width, int height, GLenum internalFormat, GLenum format, GLenum type) {
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void createFBO(GLuint& fbo, GLuint tex) {
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cerr << "Framebuffer not complete!" << std::endl;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}








// NEW: Batched splats (density or velocity) fragment shader
const char* addSourcesBatchFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D field;         // current field (RG or RGBA)
uniform sampler2D obstacles;     // obstacle mask
uniform sampler2D splatsTex;     // batched splats data
uniform int splatCount;          // number of splats in this pass
uniform vec2 texelSize;
uniform vec2 aspectRatio;

// Accumulate Gaussian splats from bullets in one pass.
// Splats are packed into a 2D RGBA32F texture, width=N, height=2:
//  row0(i) = (point.x, point.y, value.x, value.y)
//  row1(i) = (value.z, radius, _, _)

void main() {
    // Respect obstacles
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = texture(field, texCoord);
        return;
    }

    vec4 current = texture(field, texCoord);
    vec3 accum = vec3(0.0);

    // Loop over splats in this batch
    // Note: splatCount may be up to a fixed batch size (e.g., 1024)
    for (int i = 0; i < splatCount; ++i) {
        float u = (float(i) + 0.5) / float(textureSize(splatsTex, 0).x);

        // Fetch row 0 (point.xy, value.xy)
        vec4 row0 = texture(splatsTex, vec2(u, 0.25));   // y ~ 0.25 -> first row
        // Fetch row 1 (value.z, radius)
        vec4 row1 = texture(splatsTex, vec2(u, 0.75));   // y ~ 0.75 -> second row

        vec2 point = row0.xy;
        vec3 value = vec3(row0.z, row0.w, row1.x);
        float radius = row1.y;

        // Aspect-correct distance
        vec2 diff = (texCoord - point) * aspectRatio;
        float dist2 = dot(diff, diff);

        // Gaussian weight (matches your original kernel)
        float factor = exp(-dist2 / radius);

        accum += factor * value;
    }

    // Add to current field (density: accum.xy; velocity: accum.xy; third component ignored for velocity)
    fragColor = current + vec4(accum, 0.0);
}
)";



// NEW: Globals
GLuint addSourcesBatchProgram = 0;
GLuint splatsTex = 0;    // RGBA32F texture storing packed splats for the current batch
const int MAX_SPLATS_PER_PASS = 32768;







// NEW: Initialize batching resources
void initSplatBatchResources() {
	// Compile program
	addSourcesBatchProgram = createProgram(vertexShaderSource, addSourcesBatchFragmentSource);

	// Create splats texture: width = MAX_SPLATS_PER_PASS, height = 2 rows, RGBA32F
	glGenTextures(1, &splatsTex);
	glBindTexture(GL_TEXTURE_2D, splatsTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, MAX_SPLATS_PER_PASS, 2, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // precise sampling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}


// NEW: Splat struct (density or velocity splats)
struct Splat {
	// normalized [0,1] coords
	float px;
	float py;
	// value: (vx, vy, vz). For density RG use vx,vy. For velocity RG use vx,vy; vz unused.
	float vx;
	float vy;
	float vz;
	// kernel radius (same semantic as your existing addSource)
	float radius;
};

// NEW: Upload a batch and apply it in one pass
// textures/fbos/current follow your existing double-buffer pattern
void addSourcesBatch(GLuint* textures, GLuint* fbos, int& current,
	const std::vector<Splat>& splats) {
	if (splats.empty()) return;

	// We may need multiple passes if splats.size() > MAX_SPLATS_PER_PASS
	size_t offset = 0;
	while (offset < splats.size()) {
		size_t count = std::min((size_t)MAX_SPLATS_PER_PASS, splats.size() - offset);

		// Prepare CPU buffer: two rows of width 'count'
		// Row 0: (px, py, vx, vy)
		// Row 1: (vz, radius, 0, 0)
		std::vector<glm::vec4> row0(count);
		std::vector<glm::vec4> row1(count);

		for (size_t i = 0; i < count; ++i) {
			const Splat& s = splats[offset + i];
			row0[i] = glm::vec4(s.px, s.py, s.vx, s.vy);
			row1[i] = glm::vec4(s.vz, s.radius, 0.0f, 0.0f);
		}

		// Upload rows into the 2D texture via glTexSubImage2D
		glBindTexture(GL_TEXTURE_2D, splatsTex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)count, 1, GL_RGBA, GL_FLOAT, row0.data());
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 1, (GLsizei)count, 1, GL_RGBA, GL_FLOAT, row1.data());

		// Draw one fullscreen quad to apply this batch onto target field
		int dst = 1 - current;
		glBindFramebuffer(GL_FRAMEBUFFER, fbos[dst]);
		glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

		glUseProgram(addSourcesBatchProgram);

		// Bind field and obstacles
		setTextureUniform(addSourcesBatchProgram, "field", 0, textures[current]);
		setTextureUniform(addSourcesBatchProgram, "obstacles", 1, obstacleTex);
		setTextureUniform(addSourcesBatchProgram, "splatsTex", 2, splatsTex);

		glUniform1i(glGetUniformLocation(addSourcesBatchProgram, "splatCount"), (GLint)count);
		glUniform2f(glGetUniformLocation(addSourcesBatchProgram, "texelSize"),
			1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
		glUniform2f(glGetUniformLocation(addSourcesBatchProgram, "aspectRatio"),
			(float)SIM_WIDTH / SIM_HEIGHT, 1.0f);

		drawQuad();
		current = dst;

		offset += count;
	}
}




















void initShaders() {
	advectProgram = createProgram(vertexShaderSource, advectFragmentSource);
	divergenceProgram = createProgram(vertexShaderSource, divergenceFragmentSource);
	jacobiProgram = createProgram(vertexShaderSource, jacobiFragmentSource);
	gradientSubtractProgram = createProgram(vertexShaderSource, gradientSubtractFragmentSource);
	addSourceProgram = createProgram(vertexShaderSource, addSourceFragmentSource);
	boundaryProgram = createProgram(vertexShaderSource, boundaryFragmentSource);
	displayProgram = createProgram(vertexShaderSource, displayFragmentSource);
	vorticityProgram = createProgram(vertexShaderSource, vorticityFragmentSource);
	vorticityForceProgram = createProgram(vertexShaderSource, vorticityForceFragmentSource);
	obstacleProgram = createProgram(vertexShaderSource, obstacleFragmentSource);
	obstacleStampProgram = createProgram(vertexShaderSource, obstacleStampFragmentSource);
	copyProgram = createProgram(vertexShaderSource, copyFragmentSource);
	spriteProgram = createProgram(spriteVertexSource, spriteFragmentSource);
	turbulenceForceProgram = createProgram(vertexShaderSource, turbulenceForceFragmentSource);
	healthBarProgram = createProgram(healthBarVertexSource, healthBarFragmentSource);
	addSourcesBatchProgram = createProgram(vertexShaderSource, addSourcesBatchFragmentSource);

	// ============== GLOW SHADER INITIALIZATION ==============
	brightnessProgram = createProgram(vertexShaderSource, brightnessFragmentSource);
	blurProgram = createProgram(vertexShaderSource, blurFragmentSource);
	compositeProgram = createProgram(vertexShaderSource, compositeFragmentSource);
	// ============== END GLOW INITIALIZATION ==============

	// ============== CHROMATIC ABERRATION SHADER INITIALIZATION ==============
	chromaticAberrationProgram = createProgram(vertexShaderSource, chromaticAberrationFragmentSource);
	// ============== END CHROMATIC ABERRATION INITIALIZATION ==============

	waveChromaticProgram = createProgram(vertexShaderSource, waveChromaticFragmentSource);


}

void initTextures() {
	// Velocity textures (RG for x,y components)
	for (int i = 0; i < 2; i++) {
		createTexture(velocityTex[i], SIM_WIDTH, SIM_HEIGHT, GL_RG32F, GL_RG, GL_FLOAT);
		createFBO(velocityFBO[i], velocityTex[i]);
	}

	// Pressure textures
	for (int i = 0; i < 2; i++) {
		createTexture(pressureTex[i], SIM_WIDTH, SIM_HEIGHT, GL_R32F, GL_RED, GL_FLOAT);
		createFBO(pressureFBO[i], pressureTex[i]);
	}

	// Density textures (RGB for colored smoke)
	for (int i = 0; i < 2; i++) {
		createTexture(densityTex[i], SIM_WIDTH, SIM_HEIGHT, GL_RG32F, GL_RG, GL_FLOAT);
		createFBO(densityFBO[i], densityTex[i]);
	}

	// Divergence texture
	createTexture(divergenceTex, SIM_WIDTH, SIM_HEIGHT, GL_R32F, GL_RED, GL_FLOAT);
	createFBO(divergenceFBO, divergenceTex);

	// Vorticity texture
	createTexture(vorticityTex, SIM_WIDTH, SIM_HEIGHT, GL_R32F, GL_RED, GL_FLOAT);
	createFBO(vorticityFBO_obj, vorticityTex);

	// Obstacle texture
	createTexture(obstacleTex, SIM_WIDTH, SIM_HEIGHT, GL_R32F, GL_RED, GL_FLOAT);
	createFBO(obstacleFBO, obstacleTex);

	// Temp texture for boundary operations
	createTexture(tempTex, SIM_WIDTH, SIM_HEIGHT, GL_RGBA32F, GL_RGBA, GL_FLOAT);
	createFBO(tempFBO, tempTex);

	// Clear all textures
	glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < 2; i++) {
		glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[i]);
		glClear(GL_COLOR_BUFFER_BIT);
		glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[i]);
		glClear(GL_COLOR_BUFFER_BIT);
		glBindFramebuffer(GL_FRAMEBUFFER, densityFBO[i]);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void initQuad() {
	float quadVertices[] = {
		-1.0f, -1.0f,
		 1.0f, -1.0f,
		-1.0f,  1.0f,
		 1.0f,  1.0f
	};

	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);

	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

	glBindVertexArray(0);
}




void triggerWaveAtPosition(float screenX, float screenY, float spriteWidth, float spriteHeight) {
	// Convert screen position to normalized UV coordinates (0-1)
	// Center the wave on the sprite's center
	float centerX = (screenX + spriteWidth * 0.5f) / windowWidth;
	float centerY = 1.0f - (screenY + spriteHeight * 0.5f) / windowHeight; // Flip Y for UV

	// Clamp to valid range
	centerX = std::max(0.0f, std::min(1.0f, centerX));
	centerY = std::max(0.0f, std::min(1.0f, centerY));

	// Find the next available slot (circular buffer)
	waveEvents[nextWaveSlot].centerX = centerX;
	waveEvents[nextWaveSlot].centerY = centerY;
	waveEvents[nextWaveSlot].startTime = GLOBAL_TIME;
	waveEvents[nextWaveSlot].active = true;

	// Advance to next slot
	nextWaveSlot = (nextWaveSlot + 1) % MAX_WAVE_SOURCES;

	std::cout << "Wave triggered at UV(" << centerX << ", " << centerY << ")" << std::endl;
}

// Apply the wave chromatic aberration effect
void applyWaveChromaticAberration() {
	if (!waveChromaticEnabled) return;

	// Count active waves and prepare uniform data
	int activeCount = 0;
	float waveData[MAX_WAVE_SOURCES * 4]; // vec4 per source

	for (int i = 0; i < MAX_WAVE_SOURCES; i++) {
		if (waveEvents[i].active) {
			float elapsed = GLOBAL_TIME - waveEvents[i].startTime;

			// Check if wave has expired
			if (elapsed > waveDuration) {
				waveEvents[i].active = false;
				continue;
			}

			// Pack data: centerX, centerY, startTime, active
			int idx = activeCount * 4;
			waveData[idx + 0] = waveEvents[i].centerX;
			waveData[idx + 1] = waveEvents[i].centerY;
			waveData[idx + 2] = waveEvents[i].startTime;
			waveData[idx + 3] = 1.0f; // active flag
			activeCount++;
		}
	}

	// Skip rendering if no active waves
	if (activeCount == 0) return;

	// Render to screen, reading from sceneTex
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, windowWidth, windowHeight);

	glUseProgram(waveChromaticProgram);
	setTextureUniform(waveChromaticProgram, "scene", 0, sceneTex);

	// Set uniforms
	glUniform1f(glGetUniformLocation(waveChromaticProgram, "time"), GLOBAL_TIME);
	glUniform1f(glGetUniformLocation(waveChromaticProgram, "waveSpeed"), waveSpeed);
	glUniform1f(glGetUniformLocation(waveChromaticProgram, "waveDuration"), waveDuration);
	glUniform1f(glGetUniformLocation(waveChromaticProgram, "aberrationIntensity"), waveAberrationIntensity);
	glUniform1f(glGetUniformLocation(waveChromaticProgram, "ringWidth"), waveRingWidth);
	glUniform1f(glGetUniformLocation(waveChromaticProgram, "ringFalloff"), waveRingFalloff);
	glUniform1f(glGetUniformLocation(waveChromaticProgram, "aspectRatio"),
		(float)windowWidth / (float)windowHeight);

	// Upload wave source data
	glUniform4fv(glGetUniformLocation(waveChromaticProgram, "waveSources"),
		activeCount, waveData);
	glUniform1i(glGetUniformLocation(waveChromaticProgram, "activeWaveCount"), activeCount);

	drawQuad();
}



// ============== GLOW RESOURCE INITIALIZATION ==============
void initGlowResources() {
	// Scene render target (full resolution)
	createTexture(sceneTex, windowWidth, windowHeight, GL_RGBA16F, GL_RGBA, GL_FLOAT);
	createFBO(sceneFBO, sceneTex);

	// Brightness extraction (half resolution for performance)
	int glowWidth = windowWidth / 2;
	int glowHeight = windowHeight / 2;

	createTexture(brightTex, glowWidth, glowHeight, GL_RGBA16F, GL_RGBA, GL_FLOAT);
	createFBO(brightFBO, brightTex);

	// Ping-pong blur buffers (half resolution)
	for (int i = 0; i < 2; i++) {
		createTexture(blurTex[i], glowWidth, glowHeight, GL_RGBA16F, GL_RGBA, GL_FLOAT);
		createFBO(blurFBO[i], blurTex[i]);
	}

	// Clear all glow textures
	glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, brightFBO);
	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < 2; i++) {
		glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[i]);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Call this if window is resized to recreate glow buffers
void resizeGlowResources() {
	// Delete old textures and FBOs
	glDeleteTextures(1, &sceneTex);
	glDeleteFramebuffers(1, &sceneFBO);
	glDeleteTextures(1, &brightTex);
	glDeleteFramebuffers(1, &brightFBO);
	glDeleteTextures(2, blurTex);
	glDeleteFramebuffers(2, blurFBO);

	// Recreate with new dimensions
	initGlowResources();
}
// ============== END GLOW RESOURCE INITIALIZATION ==============

// ============== GLOW RENDERING FUNCTIONS ==============

// Extract bright pixels from the scene
void extractBrightPixels() {
	int glowWidth = windowWidth / 2;
	int glowHeight = windowHeight / 2;

	glBindFramebuffer(GL_FRAMEBUFFER, brightFBO);
	glViewport(0, 0, glowWidth, glowHeight);

	glUseProgram(brightnessProgram);
	setTextureUniform(brightnessProgram, "scene", 0, sceneTex);
	glUniform1f(glGetUniformLocation(brightnessProgram, "threshold"), glowThreshold);

	drawQuad();
}

// Apply Gaussian blur (horizontal + vertical = one pass)
void applyBlur() {
	int glowWidth = windowWidth / 2;
	int glowHeight = windowHeight / 2;

	int currentBuffer = 0;
	GLuint sourceTexture = brightTex;

	for (int pass = 0; pass < glowBlurPasses; pass++) {
		// Horizontal blur
		glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[currentBuffer]);
		glViewport(0, 0, glowWidth, glowHeight);

		glUseProgram(blurProgram);
		setTextureUniform(blurProgram, "image", 0, sourceTexture);
		glUniform2f(glGetUniformLocation(blurProgram, "direction"), 1.0f, 0.0f);
		glUniform2f(glGetUniformLocation(blurProgram, "texelSize"),
			1.0f / glowWidth, 1.0f / glowHeight);

		drawQuad();

		// Vertical blur
		int nextBuffer = 1 - currentBuffer;
		glBindFramebuffer(GL_FRAMEBUFFER, blurFBO[nextBuffer]);

		glUseProgram(blurProgram);
		setTextureUniform(blurProgram, "image", 0, blurTex[currentBuffer]);
		glUniform2f(glGetUniformLocation(blurProgram, "direction"), 0.0f, 1.0f);
		glUniform2f(glGetUniformLocation(blurProgram, "texelSize"),
			1.0f / glowWidth, 1.0f / glowHeight);

		drawQuad();

		// Next pass reads from the result of this pass
		sourceTexture = blurTex[nextBuffer];
		currentBuffer = nextBuffer;
	}
}

// Composite the glow with the original scene
void compositeGlow() {
	// Check if chromatic aberration will be applied
	float timeSinceDamage = GLOBAL_TIME - lastDamageTime;
	bool willApplyAberration = chromaticAberrationEnabled &&
		(timeSinceDamage <= aberrationDuration);

	// Check if wave chromatic aberration will be applied
	bool willApplyWaveAberration = false;
	if (waveChromaticEnabled) {
		for (int i = 0; i < MAX_WAVE_SOURCES; i++) {
			if (waveEvents[i].active && (GLOBAL_TIME - waveEvents[i].startTime) <= waveDuration) {
				willApplyWaveAberration = true;
				break;
			}
		}
	}

	if (willApplyAberration || willApplyWaveAberration) {
		// Render to sceneFBO so subsequent effects can read it
		glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO);
	}
	else {
		// Render directly to screen
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	glViewport(0, 0, windowWidth, windowHeight);

	glUseProgram(compositeProgram);
	setTextureUniform(compositeProgram, "scene", 0, sceneTex);

	// Use the last blur buffer (alternates based on number of passes)
	GLuint glowTexture = blurTex[(glowBlurPasses % 2 == 0) ? 1 : 0];
	if (glowBlurPasses == 0) glowTexture = brightTex;

	setTextureUniform(compositeProgram, "glow", 1, glowTexture);
	glUniform1f(glGetUniformLocation(compositeProgram, "intensity"), glowIntensity);

	drawQuad();
}

// Main glow processing pipeline
void applyGlowEffect() {
	if (!glowEnabled) return;

	extractBrightPixels();
	applyBlur();
	compositeGlow();
}
// ============== END GLOW RENDERING FUNCTIONS ==============

// ============== CHROMATIC ABERRATION RENDERING FUNCTION ==============
void applyChromaticAberration() {
	if (!chromaticAberrationEnabled) return;

	// Calculate how long since last damage
	float timeSinceDamage = GLOBAL_TIME - lastDamageTime;

	// Effect fades out over aberrationDuration seconds
	if (timeSinceDamage > aberrationDuration) return;

	// Calculate fade-out effect strength (1.0 at damage, 0.0 at end)
	float effectStrength = 1.0f - (timeSinceDamage / aberrationDuration);

	// Apply easing for smoother fade (ease-out)
	effectStrength = effectStrength * effectStrength;

	// Skip if effect is negligible
	if (effectStrength < 0.01f) return;

	// Calculate protagonist position in UV coordinates (0-1)
	// Note: Protagonist position is in screen pixels, need to convert to UV
	float protaUVx = (protagonist.x + protagonist.width * 0.5f) / windowWidth;
	float protaUVy = 1.0f - (protagonist.y + protagonist.height * 0.5f) / windowHeight; // Flip Y

	// Clamp to valid UV range
	protaUVx = std::max(0.0f, std::min(1.0f, protaUVx));
	protaUVy = std::max(0.0f, std::min(1.0f, protaUVy));

	// Check if wave chromatic will run after this
	bool waveWillRun = false;
	if (waveChromaticEnabled) {
		for (int i = 0; i < MAX_WAVE_SOURCES; i++) {
			if (waveEvents[i].active && (GLOBAL_TIME - waveEvents[i].startTime) <= waveDuration) {
				waveWillRun = true;
				break;
			}
		}
	}

	if (waveWillRun) {
		// Write to sceneFBO so wave effect can read it
		glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO);
	}
	else {
		// Write directly to screen
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}



	glViewport(0, 0, windowWidth, windowHeight);

	// Read from sceneTex (after glow composite) and write directly to screen
	glUseProgram(chromaticAberrationProgram);
	setTextureUniform(chromaticAberrationProgram, "scene", 0, sceneTex);

	glUniform2f(glGetUniformLocation(chromaticAberrationProgram, "protagonistPos"),
		protaUVx, protaUVy);
	glUniform1f(glGetUniformLocation(chromaticAberrationProgram, "intensity"),
		aberrationIntensity);
	glUniform1f(glGetUniformLocation(chromaticAberrationProgram, "effectStrength"),
		effectStrength);
	glUniform1f(glGetUniformLocation(chromaticAberrationProgram, "vignetteStrength"),
		vignetteStrength);
	glUniform1f(glGetUniformLocation(chromaticAberrationProgram, "vignetteRadius"),
		vignetteRadius);
	glUniform1f(glGetUniformLocation(chromaticAberrationProgram, "time"),
		GLOBAL_TIME);

	glUniform1f(glGetUniformLocation(chromaticAberrationProgram, "aspectRatio"),
		(float)SIM_WIDTH / (float)SIM_HEIGHT);


	drawQuad();
}
// ============== END CHROMATIC ABERRATION RENDERING FUNCTION ==============



const char* lineVertexSource = R"(
#version 400 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec4 aColor;

out vec4 vColor;

uniform vec2 resolution;  // Window dimensions

void main() {
    // Convert from pixel coordinates to NDC [-1, 1]
    vec2 ndc = (aPosition / resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y (screen coords have Y down, OpenGL has Y up)
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = aColor;
}
)";

const char* lineFragmentSource = R"(
#version 400 core
in vec4 vColor;
out vec4 fragColor;

void main() {
    fragColor = vColor;
}
)";

// ----- Point Shader Sources -----

const char* pointVertexSource = R"(
#version 400 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec4 aColor;

out vec4 vColor;

uniform vec2 resolution;  // Window dimensions

void main() {
    // Convert from pixel coordinates to NDC [-1, 1]
    vec2 ndc = (aPosition / resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y (screen coords have Y down, OpenGL has Y up)
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = aColor;
}
)";

const char* pointFragmentSource = R"(
#version 400 core
in vec4 vColor;
out vec4 fragColor;

void main() {
    fragColor = vColor;
}
)";

// ----- Data Structures -----

struct LineVertex {
	glm::vec2 position;
	glm::vec4 color;
};

struct Line {
	glm::vec2 start;
	glm::vec2 end;
	glm::vec4 color;  // RGBA, values 0.0-1.0

	Line(glm::vec2 s, glm::vec2 e, glm::vec4 c)
		: start(s), end(e), color(c) {
	}

	// Convenience constructor with default white color
	Line(glm::vec2 s, glm::vec2 e)
		: start(s), end(e), color(1.0f, 1.0f, 1.0f, 1.0f) {
	}
};






GLuint lineProgram = 0;
GLuint lineVAO = 0;
GLuint lineVBO = 0;
std::vector<Line> lines;  // Your vector of lines

vector<Line> tangent_lines;


// ----- Point Data Structure -----

struct PointVertex {
	glm::vec2 position;
	glm::vec4 color;
};

struct Point {
	glm::vec2 position;
	glm::vec4 color;  // RGBA, values 0.0-1.0

	Point(glm::vec2 p, glm::vec4 c)
		: position(p), color(c) {
	}

	Point(float x, float y, glm::vec4 c)
		: position(x, y), color(c) {
	}

	// Convenience constructor with default white color
	Point(glm::vec2 p)
		: position(p), color(1.0f, 1.0f, 1.0f, 1.0f) {
	}

	Point(float x, float y)
		: position(x, y), color(1.0f, 1.0f, 1.0f, 1.0f) {
	}
};

GLuint pointProgram = 0;
GLuint pointVAO = 0;
GLuint pointVBO = 0;
std::vector<Point> points;  // Your vector of points


void initLineRenderer() {
	// Create shader program
	lineProgram = createProgram(lineVertexSource, lineFragmentSource);

	// Create VAO and VBO
	glGenVertexArrays(1, &lineVAO);
	glGenBuffers(1, &lineVBO);

	glBindVertexArray(lineVAO);
	glBindBuffer(GL_ARRAY_BUFFER, lineVBO);

	// Position attribute (location 0)
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(LineVertex),
		(void*)offsetof(LineVertex, position));

	// Color attribute (location 1)
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(LineVertex),
		(void*)offsetof(LineVertex, color));

	glBindVertexArray(0);
}

// ----- Drawing -----

void drawLines(const std::vector<Line>& linesToDraw) {
	if (linesToDraw.empty()) return;

	// Build vertex data from lines
	std::vector<LineVertex> vertices;
	vertices.reserve(linesToDraw.size() * 2);

	for (const auto& line : linesToDraw) {
		vertices.push_back({ line.start, line.color });
		vertices.push_back({ line.end, line.color });
	}

	// Upload vertex data
	glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
	glBufferData(GL_ARRAY_BUFFER,
		vertices.size() * sizeof(LineVertex),
		vertices.data(),
		GL_DYNAMIC_DRAW);

	glEnable(GL_BLEND);

	// Draw
	glUseProgram(lineProgram);
	glUniform2f(glGetUniformLocation(lineProgram, "resolution"),
		(float)windowWidth, (float)windowHeight);

	glBindVertexArray(lineVAO);
	glDrawArrays(GL_LINES, 0, (GLsizei)vertices.size());
	glBindVertexArray(0);

	glDisable(GL_BLEND);

}

// Optional: Draw with custom line width
void drawLinesWithWidth(const std::vector<Line>& linesToDraw, float width) {
	glLineWidth(width);
	drawLines(linesToDraw);
	glLineWidth(1.0f);  // Reset to default
}

// ============== POINT RENDERER ==============

void initPointRenderer() {
	// Create shader program (uses same simple vertex/fragment shaders as lines)
	pointProgram = createProgram(pointVertexSource, pointFragmentSource);

	// Create VAO and VBO
	glGenVertexArrays(1, &pointVAO);
	glGenBuffers(1, &pointVBO);

	glBindVertexArray(pointVAO);
	glBindBuffer(GL_ARRAY_BUFFER, pointVBO);

	// Position attribute (location 0)
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(PointVertex),
		(void*)offsetof(PointVertex, position));

	// Color attribute (location 1)
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(PointVertex),
		(void*)offsetof(PointVertex, color));

	glBindVertexArray(0);

	// Enable point size control from vertex shader (optional, for gl_PointSize)
	glEnable(GL_PROGRAM_POINT_SIZE);
}

// ----- Point Drawing -----

void drawPoints(const std::vector<Point>& pointsToDraw) {
	if (pointsToDraw.empty()) return;

	// Build vertex data from points
	std::vector<PointVertex> vertices;
	vertices.reserve(pointsToDraw.size());

	for (const auto& point : pointsToDraw) {
		vertices.push_back({ point.position, point.color });
	}

	// Upload vertex data
	glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
	glBufferData(GL_ARRAY_BUFFER,
		vertices.size() * sizeof(PointVertex),
		vertices.data(),
		GL_DYNAMIC_DRAW);

	// Draw
	glUseProgram(pointProgram);
	glUniform2f(glGetUniformLocation(pointProgram, "resolution"),
		(float)windowWidth, (float)windowHeight);

	glBindVertexArray(pointVAO);
	glDrawArrays(GL_POINTS, 0, (GLsizei)vertices.size());
	glBindVertexArray(0);
}

// Draw with custom point size
void drawPointsWithSize(const std::vector<Point>& pointsToDraw, float size) {
	glPointSize(size);
	drawPoints(pointsToDraw);
	glPointSize(1.0f);  // Reset to default
}

// Draw a single point (convenience function)
void drawPoint(const Point& point) {
	std::vector<Point> singlePoint = { point };
	drawPoints(singlePoint);
}

// Draw a single point with size (convenience function)
void drawPointWithSize(const Point& point, float size) {
	std::vector<Point> singlePoint = { point };
	drawPointsWithSize(singlePoint, size);
}

// Draw a single point at x,y with color (convenience function)
void drawPoint(float x, float y, glm::vec4 color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)) {
	std::vector<Point> singlePoint = { Point(x, y, color) };
	drawPoints(singlePoint);
}

// Draw a single point at x,y with size and color (convenience function)
void drawPointWithSize(float x, float y, float size, glm::vec4 color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)) {
	std::vector<Point> singlePoint = { Point(x, y, color) };
	drawPointsWithSize(singlePoint, size);
}

// ============== END POINT RENDERER ==============


void initCollisionResources()
{
	// 2-channel floating point texture: R = red density, G = green density
	glGenTextures(1, &collisionTex);
	glBindTexture(GL_TEXTURE_2D, collisionTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, SIM_WIDTH, SIM_HEIGHT, 0, GL_RG, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glGenFramebuffers(1, &collisionFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, collisionFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, collisionTex, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cerr << "Collision FBO not complete!" << std::endl;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Compile shader
	collisionProgram = createProgram(vertexShaderSource, collisionFragmentSource);
}









/**
 * Test if a screen pixel coordinate lies inside a sprite, and if so,
 * whether the corresponding sprite texel is transparent.
 *
 * @param sprTex       OpenGL texture ID of the sprite (RGBA8 or similar)
 * @param sprX         Sprite top-left corner X in window pixels
 * @param sprY         Sprite top-left corner Y in window pixels
 * @param sprW         Sprite width in pixels
 * @param sprH         Sprite height in pixels
 * @param pixelX       Test point X in window pixels
 * @param pixelY       Test point Y in window pixels
 * @param outInside    Output: true if inside sprite bounds
 * @param outTransparent Output: true if the sprite pixel is transparent (alpha < threshold)
 *
 * NOTE: This reads the entire sprite texture into CPU memory ONCE per call.
 * If needed often, cache the pixel buffer and width/height externally.
 */
bool isPixelInsideTriSpriteAndTransparent(
	tri_sprite& spr,
	GLuint sprTex,
	int sprX, int sprY,
	int sprW, int sprH,
	int pixelX, int pixelY,
	bool& outInside,
	bool& outTransparent,
	unsigned char alphaThreshold,
	glm::vec2& hit)
{
	outInside = false;
	outTransparent = false;
	hit = glm::vec2(0, 0);

	// 1. Bounding box test (fast)
	if (pixelX < sprX || pixelX >= sprX + sprW ||
		pixelY < sprY || pixelY >= sprY + sprH)
	{
		return false; // definitely not inside
	}

	outInside = true;

	// 2. Compute sprite-relative pixel coordinates
	int localX = pixelX - sprX;
	int localY = pixelY - sprY;

	// Sprite textures use bottom-left origin by default.
	// Screen pixels use top-left origin.
	// Convert Y accordingly.
	int texY = (sprH - 1) - localY;
	int texX = localX;

	// 3. Read texture pixel data from GPU
	//    (4 bytes per pixel: RGBA8)
	//std::vector<unsigned char> texData(sprW * sprH * 4);

	//glBindTexture(GL_TEXTURE_2D, sprTex);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, texData.data());


	unsigned char* data_ptr = 0;

	// Use the active frame based on current state (n-sprite aware)
	{
		int idx = spr.state;
		if (idx < 0) idx = 0;
		if (idx >= (int)spr.sprite_frames.size()) idx = (int)spr.sprite_frames.size() - 1;

		if (!spr.sprite_frames.empty() && !spr.sprite_frames[idx].empty())
			data_ptr = const_cast<unsigned char*>(spr.sprite_frames[idx].data());
		else if (!spr.sprite_frames.empty() && !spr.sprite_frames[spr.rest_state_index()].empty())
			data_ptr = const_cast<unsigned char*>(spr.sprite_frames[spr.rest_state_index()].data());
	}



	// 4. Index into pixel buffer
	int idx = (texY * sprW + texX) * 4;
	unsigned char a = data_ptr[idx + 3];  // ALPHA

	// 5. Transparent?
	outTransparent = (a < alphaThreshold);

	hit = glm::vec2(localX, localY);

	return true;
}




bool isPixelInsideSpriteAndTransparent(
	sprite& spr,
	GLuint sprTex,
	int sprX, int sprY,
	int sprW, int sprH,
	int pixelX, int pixelY,
	bool& outInside,
	bool& outTransparent,
	unsigned char alphaThreshold,
	glm::vec2& hit)
{
	outInside = false;
	outTransparent = false;
	hit = glm::vec2(0, 0);

	// 1. Bounding box test (fast)
	if (pixelX < sprX || pixelX >= sprX + sprW ||
		pixelY < sprY || pixelY >= sprY + sprH)
	{
		return false; // definitely not inside
	}

	outInside = true;

	// 2. Compute sprite-relative pixel coordinates
	int localX = pixelX - sprX;
	int localY = pixelY - sprY;

	// Sprite textures use bottom-left origin by default.
	// Screen pixels use top-left origin.
	// Convert Y accordingly.
	int texY = (sprH - 1) - localY;
	int texX = localX;

	// 3. Read texture pixel data from GPU
	//    (4 bytes per pixel: RGBA8)
	//std::vector<unsigned char> texData(sprW * sprH * 4);

	//glBindTexture(GL_TEXTURE_2D, sprTex);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, texData.data());


	unsigned char* data_ptr = spr.to_present_data.data();

	// 4. Index into pixel buffer
	int idx = (texY * sprW + texX) * 4;
	unsigned char a = data_ptr[idx + 3];  // ALPHA

	// 5. Transparent?
	outTransparent = (a < alphaThreshold);

	hit = glm::vec2(localX, localY);

	return true;
}





void applyTurbulence()
{
	int dst = 1 - currentVelocity;
	glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[dst]);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

	glUseProgram(turbulenceForceProgram);

	// Set uniforms
	glUniform1f(glGetUniformLocation(turbulenceForceProgram, "time"), GLOBAL_TIME);
	glUniform1f(glGetUniformLocation(turbulenceForceProgram, "amplitude"), TURBULENCE_AMPLITUDE);
	glUniform1f(glGetUniformLocation(turbulenceForceProgram, "frequency"), TURBULENCE_FREQUENCY);
	glUniform1f(glGetUniformLocation(turbulenceForceProgram, "scale"), TURBULENCE_SCALE * DT); // Scale by timestep

	// Set texture uniforms
	setTextureUniform(turbulenceForceProgram, "velocity", 0, velocityTex[currentVelocity]);
	setTextureUniform(turbulenceForceProgram, "obstacles", 1, obstacleTex);
	glUniform2f(glGetUniformLocation(turbulenceForceProgram, "texelSize"),
		1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);

	drawQuad();
	currentVelocity = dst;
}



void detectEdgeCollisions()
{
	collisionPoints.clear();

	// Step 1: Render collision map (edge + density values)
	glBindFramebuffer(GL_FRAMEBUFFER, collisionFBO);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(collisionProgram);
	setTextureUniform(collisionProgram, "density", 0, densityTex[currentDensity]);
	setTextureUniform(collisionProgram, "obstacles", 1, obstacleTex);
	glUniform2f(glGetUniformLocation(collisionProgram, "texelSize"),
		1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);

	drawQuad();

	// Step 2: Read back the RG32F texture
	std::vector<glm::vec2> pixelData(SIM_WIDTH * SIM_HEIGHT);
	glBindFramebuffer(GL_FRAMEBUFFER, collisionFBO);
	glReadPixels(0, 0, SIM_WIDTH, SIM_HEIGHT, GL_RG, GL_FLOAT, pixelData.data());


	float target = 1.0;

	// Step 3: Collect all non-zero collision points
	for (int y = 0; y < SIM_HEIGHT; ++y)
	{
		for (int x = 0; x < SIM_WIDTH; ++x)
		{
			size_t idx = y * SIM_WIDTH + x;
			glm::vec2 dens = pixelData[idx];

			// If either red or green density is present -> collision
			if (dens.r >= target || dens.g >= target)
			{
				if (dens.r > target)
					dens.r = target;

				if (dens.g > target)
					dens.g = target;

				collisionPoints.push_back(glm::vec4(
					static_cast<float>(x),
					static_cast<float>(SIM_HEIGHT - 1 - y),
					dens.r,   // red density
					dens.g    // green density
				));
			}
		}
	}



	{
		if (1)//collisionPoints.size() > 0)
		{
			protagonist.under_fire = false;
			vector<glm::vec2> protagonist_blackening_points;

			for (size_t i = 0; i < collisionPoints.size(); i++)
			{
				//cout << collisionPoints[i].x << " " << collisionPoints[i].y << endl;
				//cout << collisionPoints[i].z << " " << collisionPoints[i].w << endl;



				bool inside = false, transparent = false;



				glm::vec2 hit;

				if (isPixelInsideTriSpriteAndTransparent(
					protagonist,
					protagonist.tex,
					static_cast<int>(protagonist.x),
					static_cast<int>(protagonist.y),
					protagonist.width,
					protagonist.height,
					static_cast<int>(collisionPoints[i].x),
					static_cast<int>(collisionPoints[i].y),
					inside,
					transparent,
					127,
					hit))
				{
					if (inside && collisionPoints[i].w >= target)
					{
						const float DAMAGE_INTERVAL = aberrationDuration;  // seconds between damage ticks

						if (GLOBAL_TIME - protagonist.last_time_collided >= DAMAGE_INTERVAL)
						{
							protagonist.health -= 100.0;// collisionPoints[i].w;
							protagonist.under_fire = true;
							protagonist_blackening_points.push_back(glm::vec2(hit.x, hit.y));

							protagonist.last_time_collided = GLOBAL_TIME;

							// Trigger chromatic aberration effect on collision damage
							lastDamageTime = GLOBAL_TIME;
						}
					}
				}
			}

			protagonist.animate_blackening(protagonist_blackening_points, protagonist.state);
		}


		dis_real.reset();

		for (size_t h = 0; h < foreground_chunked.size(); h++)
		{
			vector<glm::vec2> blackening_points;

			for (size_t i = 0; i < collisionPoints.size(); i++)
			{
				//cout << collisionPoints[i].x << " " << collisionPoints[i].y << endl;
				//cout << collisionPoints[i].z << " " << collisionPoints[i].w << endl;



				bool inside = false, transparent = false;



				glm::vec2 hit;

				if (isPixelInsideSpriteAndTransparent(
					foreground_chunked[h],
					foreground_chunked[h].tex,
					static_cast<int>(foreground_chunked[h].x),
					static_cast<int>(foreground_chunked[h].y),
					foreground_chunked[h].width,
					foreground_chunked[h].height,
					static_cast<int>(collisionPoints[i].x),
					static_cast<int>(collisionPoints[i].y),
					inside,
					transparent,
					127,
					hit))
				{
					if (inside)
					{
						blackening_points.push_back(glm::vec2(hit.x, hit.y));
					}
				}
			}

			foreground_chunked[h].animate_blackening(blackening_points, 0);
		}


		dis_real.reset();

		for (size_t h = 0; h < foreground_lit_chunked.size(); h++)
		{
			vector<glm::vec2> blackening_points;

			for (size_t i = 0; i < collisionPoints.size(); i++)
			{
				bool inside = false, transparent = false;
				glm::vec2 hit;

				if (isPixelInsideSpriteAndTransparent(
					foreground_lit_chunked[h],
					foreground_lit_chunked[h].tex,
					static_cast<int>(foreground_lit_chunked[h].x),
					static_cast<int>(foreground_lit_chunked[h].y),
					foreground_lit_chunked[h].width,
					foreground_lit_chunked[h].height,
					static_cast<int>(collisionPoints[i].x),
					static_cast<int>(collisionPoints[i].y),
					inside,
					transparent,
					127,
					hit))
				{
					if (inside)
					{
						blackening_points.push_back(glm::vec2(hit.x, hit.y));
					}
				}
			}

			foreground_lit_chunked[h].animate_blackening(blackening_points, 0);
		}


		for (size_t h = 0; h < enemy_ships.size(); h++)

		{
			enemy_ships[h]->under_fire = false;
			vector<glm::vec2> blackening_points;

			for (size_t i = 0; i < collisionPoints.size(); i++)
			{
				//cout << collisionPoints[i].x << " " << collisionPoints[i].y << endl;
				//cout << collisionPoints[i].z << " " << collisionPoints[i].w << endl;



				bool inside = false, transparent = false;



				glm::vec2 hit;

				if (isPixelInsideTriSpriteAndTransparent(
					*enemy_ships[h],
					enemy_ships[h]->tex,
					static_cast<int>(enemy_ships[h]->x),
					static_cast<int>(enemy_ships[h]->y),
					enemy_ships[h]->width,
					enemy_ships[h]->height,
					static_cast<int>(collisionPoints[i].x),
					static_cast<int>(collisionPoints[i].y),
					inside,
					transparent,
					127,
					hit))
				{
					if (inside && collisionPoints[i].z >= target)
					{
						enemy_ships[h]->health -= collisionPoints[i].z;
						enemy_ships[h]->under_fire = true;
						blackening_points.push_back(glm::vec2(hit.x, hit.y));
					}
				}
			}

			if (false == enemy_ships[h]->to_be_culled)
				enemy_ships[h]->blackenChunks(blackening_points);
		}
	}
}



void advect(GLuint velocityTex, GLuint quantityTex, GLuint outputFBO, float dissipation) {
	glBindFramebuffer(GL_FRAMEBUFFER, outputFBO);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

	glUseProgram(advectProgram);
	setTextureUniform(advectProgram, "velocity", 0, velocityTex);
	setTextureUniform(advectProgram, "quantity", 1, quantityTex);
	setTextureUniform(advectProgram, "obstacles", 2, obstacleTex);
	glUniform2f(glGetUniformLocation(advectProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
	glUniform1f(glGetUniformLocation(advectProgram, "dt"), DT * 100.0f);
	glUniform1f(glGetUniformLocation(advectProgram, "dissipation"), dissipation);

	drawQuad();
}

void computeDivergence() {
	glBindFramebuffer(GL_FRAMEBUFFER, divergenceFBO);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

	glUseProgram(divergenceProgram);
	setTextureUniform(divergenceProgram, "velocity", 0, velocityTex[currentVelocity]);
	setTextureUniform(divergenceProgram, "obstacles", 1, obstacleTex);
	glUniform2f(glGetUniformLocation(divergenceProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);

	drawQuad();
}

void clearPressure() {
	glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[currentPressure]);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
}

void jacobi() {
	glUseProgram(jacobiProgram);
	glUniform2f(glGetUniformLocation(jacobiProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
	glUniform1f(glGetUniformLocation(jacobiProgram, "alpha"), -1.0f);
	glUniform1f(glGetUniformLocation(jacobiProgram, "rBeta"), 0.25f);

	for (int i = 0; i < JACOBI_ITERATIONS; i++) {
		int dst = 1 - currentPressure;
		glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[dst]);
		glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

		setTextureUniform(jacobiProgram, "pressure", 0, pressureTex[currentPressure]);
		setTextureUniform(jacobiProgram, "divergence", 1, divergenceTex);
		setTextureUniform(jacobiProgram, "obstacles", 2, obstacleTex);

		drawQuad();
		currentPressure = dst;
	}
}

void subtractGradient() {
	int dst = 1 - currentVelocity;
	glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[dst]);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

	glUseProgram(gradientSubtractProgram);
	setTextureUniform(gradientSubtractProgram, "pressure", 0, pressureTex[currentPressure]);
	setTextureUniform(gradientSubtractProgram, "velocity", 1, velocityTex[currentVelocity]);
	setTextureUniform(gradientSubtractProgram, "obstacles", 2, obstacleTex);
	glUniform2f(glGetUniformLocation(gradientSubtractProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);

	drawQuad();
	currentVelocity = dst;
}

void computeVorticity() {
	glBindFramebuffer(GL_FRAMEBUFFER, vorticityFBO_obj);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

	glUseProgram(vorticityProgram);
	setTextureUniform(vorticityProgram, "velocity", 0, velocityTex[currentVelocity]);
	setTextureUniform(vorticityProgram, "obstacles", 1, obstacleTex);
	glUniform2f(glGetUniformLocation(vorticityProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);

	drawQuad();
}

void applyVorticityForce() {
	int dst = 1 - currentVelocity;
	glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[dst]);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

	glUseProgram(vorticityForceProgram);
	setTextureUniform(vorticityForceProgram, "velocity", 0, velocityTex[currentVelocity]);
	setTextureUniform(vorticityForceProgram, "vorticity", 1, vorticityTex);
	setTextureUniform(vorticityForceProgram, "obstacles", 2, obstacleTex);
	glUniform2f(glGetUniformLocation(vorticityForceProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
	glUniform1f(glGetUniformLocation(vorticityForceProgram, "dt"), DT);
	glUniform1f(glGetUniformLocation(vorticityForceProgram, "scale"), VORTICITY_SCALE);

	drawQuad();
	currentVelocity = dst;
}

//void applyTurbulence()
//{
//	glUseProgram(turbulenceForceProgram);
//
//	glUniform1f(glGetUniformLocation(turbulenceForceProgram, "time"), GLOBAL_TIME);
//	glUniform1f(glGetUniformLocation(turbulenceForceProgram, "amplitude"), TURBULENCE_AMPLITUDE);
//	glUniform1f(glGetUniformLocation(turbulenceForceProgram, "frequency"), TURBULENCE_FREQUENCY);
//	glUniform1f(glGetUniformLocation(turbulenceForceProgram, "scale"), TURBULENCE_SCALE);
//
//	glUniform2f(glGetUniformLocation(turbulenceForceProgram, "texelSize"),
//		1.0f / SIM_WIDTH,
//		1.0f / SIM_HEIGHT);
//
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, velocityTex[currentVelocity]);
//	glUniform1i(glGetUniformLocation(turbulenceForceProgram, "velocity"), 0);
//
//	glActiveTexture(GL_TEXTURE1);
//	glBindTexture(GL_TEXTURE_2D, obstacleTex);
//	glUniform1i(glGetUniformLocation(turbulenceForceProgram, "obstacles"), 1);
//
//	int nextVelocity = 1 - currentVelocity;
//	glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[nextVelocity]);
//
//	glBindVertexArray(quadVAO);
//	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
//
//	currentVelocity = nextVelocity;
//}

void addSource(GLuint* textures, GLuint* fbos, int& current, float x, float y, float vx, float vy, float vz, float radius)
{
	int dst = 1 - current;
	glBindFramebuffer(GL_FRAMEBUFFER, fbos[dst]);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

	glUseProgram(addSourceProgram);
	setTextureUniform(addSourceProgram, "field", 0, textures[current]);
	setTextureUniform(addSourceProgram, "obstacles", 1, obstacleTex);
	glUniform2f(glGetUniformLocation(addSourceProgram, "point"), x, y);
	glUniform3f(glGetUniformLocation(addSourceProgram, "value"), vx, vy, vz);
	glUniform1f(glGetUniformLocation(addSourceProgram, "radius"), radius);
	glUniform2f(glGetUniformLocation(addSourceProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
	// Account for aspect ratio in splat
	glUniform2f(glGetUniformLocation(addSourceProgram, "aspectRatio"), (float)SIM_WIDTH / SIM_HEIGHT, 1.0f);

	drawQuad();
	current = dst;
}

//
//void addObstacle(float x, float y, float radius, bool add) {
//	glBindFramebuffer(GL_FRAMEBUFFER, tempFBO);
//	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);
//
//	glUseProgram(obstacleProgram);
//	setTextureUniform(obstacleProgram, "obstacles", 0, obstacleTex);
//	glUniform2f(glGetUniformLocation(obstacleProgram, "point"), x, y);
//	glUniform1f(glGetUniformLocation(obstacleProgram, "radius"), radius);
//	glUniform2f(glGetUniformLocation(obstacleProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
//	glUniform2f(glGetUniformLocation(obstacleProgram, "aspectRatio"), (float)SIM_WIDTH / SIM_HEIGHT, 1.0f);
//	glUniform1f(glGetUniformLocation(obstacleProgram, "addOrRemove"), add ? 1.0f : 0.0f);
//
//	drawQuad();
//
//	// Copy back to obstacle texture
//	glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
//	glUseProgram(copyProgram);
//	setTextureUniform(copyProgram, "source", 0, tempTex);
//	drawQuad();
//}

void addObstacleStamp(GLuint stampTexture, int pixelX, int pixelY,
	int pixelWidth, int pixelHeight, bool add,
	float threshold = 0.5f, bool useAlpha = true) {
	glBindFramebuffer(GL_FRAMEBUFFER, tempFBO);
	glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

	glUseProgram(obstacleStampProgram);

	// Bind textures
	setTextureUniform(obstacleStampProgram, "obstacles", 0, obstacleTex);
	setTextureUniform(obstacleStampProgram, "stampTexture", 1, stampTexture);

	// Convert pixel coordinates to normalized texture coordinates
	// Input: top-left origin (0,0), Y increases downward
	// OpenGL texture coords: bottom-left origin (0,0), Y increases upward

	// Calculate center position from top-left corner
	// Center in pixel coords (top-left origin)
	float centerPixelX = pixelX + pixelWidth / 2.0f;
	float centerPixelY = pixelY + pixelHeight / 2.0f;

	// Convert to normalized coords [0,1] with bottom-left origin
	// Flip Y: OpenGL's Y=0 is at bottom, our input Y=0 is at top
	float centerNormX = centerPixelX / (float)SIM_WIDTH;
	float centerNormY = 1.0f - (centerPixelY / (float)SIM_HEIGHT);

	// Half-size in normalized coords
	float halfSizeX = (float)pixelWidth / (2.0f * SIM_WIDTH);
	float halfSizeY = (float)pixelHeight / (2.0f * SIM_HEIGHT);

	// Set uniforms
	glUniform2f(glGetUniformLocation(obstacleStampProgram, "stampCenter"), centerNormX, centerNormY);
	glUniform2f(glGetUniformLocation(obstacleStampProgram, "stampHalfSize"), halfSizeX, halfSizeY);
	glUniform1f(glGetUniformLocation(obstacleStampProgram, "threshold"), threshold);
	glUniform1f(glGetUniformLocation(obstacleStampProgram, "addOrRemove"), add ? 1.0f : 0.0f);
	glUniform1i(glGetUniformLocation(obstacleStampProgram, "useAlpha"), useAlpha ? 1 : 0);

	drawQuad();

	// Copy back to obstacle texture
	glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
	glUseProgram(copyProgram);
	setTextureUniform(copyProgram, "source", 0, tempTex);
	drawQuad();
}

/**
 * Helper function to load a texture from raw RGBA pixel data.
 * Useful for creating stamp textures programmatically or from loaded images.
 *
 * @param data    Pointer to RGBA pixel data (4 bytes per pixel)
 * @param width   Width of the texture in pixels
 * @param height  Height of the texture in pixels
 * @return        OpenGL texture ID
 */
GLuint createStampTextureFromData(const unsigned char* data, int width, int height) {
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	return tex;
}

/**
 * Create a circular stamp texture programmatically.
 *
 * @param radius  Radius in pixels
 * @return        OpenGL texture ID of a circle stamp
 */
GLuint createCircleStamp(int radius) {
	int size = radius * 2;
	std::vector<unsigned char> data(size * size * 4, 0);

	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			float dx = x - radius + 0.5f;
			float dy = y - radius + 0.5f;
			float dist = sqrtf(dx * dx + dy * dy);

			int idx = (y * size + x) * 4;
			if (dist <= radius) {
				data[idx + 0] = 255;  // R
				data[idx + 1] = 255;  // G
				data[idx + 2] = 255;  // B
				data[idx + 3] = 255;  // A
			}
		}
	}

	return createStampTextureFromData(data.data(), size, size);
}

/**
 * Create a star-shaped stamp texture programmatically.
 *
 * @param outerRadius  Outer radius of star points in pixels
 * @param innerRadius  Inner radius (between points) in pixels
 * @param points       Number of star points
 * @return             OpenGL texture ID of a star stamp
 */
GLuint createStarStamp(int outerRadius, int innerRadius, int points) {
	int size = outerRadius * 2;
	std::vector<unsigned char> data(size * size * 4, 0);

	float angleStep = 3.14159265f / points;

	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			float dx = x - outerRadius + 0.5f;
			float dy = y - outerRadius + 0.5f;
			float dist = sqrtf(dx * dx + dy * dy);
			float angle = atan2f(dy, dx);
			if (angle < 0) angle += 2.0f * 3.14159265f;

			// Calculate the radius at this angle for the star shape
			int sector = (int)(angle / angleStep);
			float localAngle = fmodf(angle, angleStep * 2.0f);
			if (localAngle > angleStep) localAngle = angleStep * 2.0f - localAngle;

			float starRadius = innerRadius + (outerRadius - innerRadius) * (1.0f - localAngle / angleStep);

			int idx = (y * size + x) * 4;
			if (dist <= starRadius) {
				data[idx + 0] = 255;
				data[idx + 1] = 255;
				data[idx + 2] = 255;
				data[idx + 3] = 255;
			}
		}
	}

	return createStampTextureFromData(data.data(), size, size);
}

/**
 * Create a rectangular stamp texture.
 *
 * @param width   Width in pixels
 * @param height  Height in pixels
 * @return        OpenGL texture ID of a rectangle stamp
 */
GLuint createRectangleStamp(int width, int height) {
	std::vector<unsigned char> data(width * height * 4, 255);  // All white, fully opaque
	return createStampTextureFromData(data.data(), width, height);
}

/**
 * Load a texture from an image file using stb_image.
 * Supports PNG, JPG, BMP, TGA, and other common formats.
 *
 * @param filename    Path to the image file
 * @param outWidth    Output parameter for the image width in pixels
 * @param outHeight   Output parameter for the image height in pixels
 * @return            OpenGL texture ID, or 0 if loading failed
 */
GLuint loadTextureFromFile(const char* filename, int* outWidth, int* outHeight, vector<unsigned char>& out_data, bool repeat_texture = false) {
	int width, height, channels;

	out_data.clear();

	// stb_image loads with (0,0) at top-left, which matches our coordinate system
	stbi_set_flip_vertically_on_load(0);  // Don't flip - we handle it in the shader

	unsigned char* data = stbi_load(filename, &width, &height, &channels, 4);  // Force RGBA
	if (!data) {
		std::cerr << "Failed to load texture: " << filename << std::endl;
		std::cerr << "stb_image error: " << stbi_failure_reason() << std::endl;
		if (outWidth) *outWidth = 0;
		if (outHeight) *outHeight = 0;
		return 0;
	}

	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


	if (repeat_texture)
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	}
	else
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	for (size_t i = 0; i < width * height * channels; i++)
		out_data.push_back(data[i]);

	stbi_image_free(data);

	if (outWidth) *outWidth = width;
	if (outHeight) *outHeight = height;

	std::cout << "Loaded texture: " << filename << " (" << width << "x" << height << ")" << std::endl;
	return tex;
}



// Load n sprite frame files into a tri_sprite. The filenames vector must have
// an odd number of entries, ordered from the "most up" frame to the "most down"
// frame, with the middle entry being the rest frame.
GLuint loadTextureFromFile_NSprite(
	const std::vector<std::string>& filenames,
	int* outWidth, int* outHeight,
	tri_sprite& t)
{
	if (filenames.empty() || filenames.size() % 2 == 0)
	{
		std::cerr << "loadTextureFromFile_NSprite: need an odd number of frame files (got "
			<< filenames.size() << ")" << std::endl;
		if (outWidth) *outWidth = 0;
		if (outHeight) *outHeight = 0;
		return 0;
	}

	int width = 0, height = 0, channels = 0;

	t.sprite_frames.clear();
	t.sprite_frames.resize(filenames.size());

	stbi_set_flip_vertically_on_load(0);

	unsigned char* rest_raw = nullptr; // we keep a pointer to the rest frame for the initial texture upload

	for (size_t f = 0; f < filenames.size(); f++)
	{
		int w, h, c;
		unsigned char* data = stbi_load(filenames[f].c_str(), &w, &h, &c, 4);
		if (!data)
		{
			std::cerr << "Failed to load texture: " << filenames[f] << std::endl;
			std::cerr << "stb_image error: " << stbi_failure_reason() << std::endl;
			// Free any rest_raw we were keeping
			// (all previous frames already copied & freed)
			if (outWidth) *outWidth = 0;
			if (outHeight) *outHeight = 0;
			return 0;
		}

		if (f == 0)
		{
			width = w;
			height = h;
			channels = 4; // forced RGBA
		}

		t.sprite_frames[f].assign(data, data + (size_t)w * h * 4);

		// Keep the rest frame raw pointer for the initial GL upload, free later
		if ((int)f == (int)filenames.size() / 2)
		{
			rest_raw = data; // don't free yet
		}
		else
		{
			stbi_image_free(data);
		}
	}

	t.original_sprite_frames = t.sprite_frames;

	// Create GL texture using the rest frame
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rest_raw);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	stbi_image_free(rest_raw);

	if (outWidth) *outWidth = width;
	if (outHeight) *outHeight = height;

	t.state = t.rest_state_index();
	t.rebuild_pointers();

	return tex;
}
//
//// Legacy wrapper: load exactly 3 files (up, down, rest) in the old calling convention.
//// Internally reorders to n-sprite order: [up, rest, down] (indices 0, 1, 2).
//GLuint loadTextureFromFile_Triplet(
//	const char* up_filename,
//	const char* down_filename,
//	const char* rest_filename,
//	int* outWidth, int* outHeight,
//	vector<unsigned char>& out_up_data,
//	vector<unsigned char>& out_down_data,
//	vector<unsigned char>& out_rest_data,
//	tri_sprite& t)
//{
//	std::vector<std::string> filenames = {
//		up_filename,
//		rest_filename,
//		down_filename
//	};
//
//	GLuint tex = loadTextureFromFile_NSprite(filenames, outWidth, outHeight, t);
//
//	// Copy frame data back to the legacy output vectors for callers that still need them.
//	if (tex != 0 && t.sprite_frames.size() == 3)
//	{
//		out_up_data = t.sprite_frames[0];
//		out_rest_data = t.sprite_frames[1];
//		out_down_data = t.sprite_frames[2];
//	}
//
//	return tex;
//}




/**
 * Chunk the foreground texture into tiles of foreground_chunk_size x foreground_chunk_size pixels.
 * Each tile is stored as a separate OpenGL texture in the foreground_chunked vector.
 *
 * The tiles are arranged in row-major order:
 *   - First row: tiles[0], tiles[1], tiles[2], ...
 *   - Second row: tiles[cols], tiles[cols+1], ...
 *
 * Each tile's x,y position is set to its position in the original image layout.
 *
 * @param sourceFilename  Path to the foreground image file
 * @return                True if chunking succeeded, false otherwise
 */
bool chunkForegroundTexture(const char* sourceFilename, vector<foreground_tile>& target = foreground_chunked)
{
	target.clear();

	int srcWidth, srcHeight, channels;

	// Load the source image
	stbi_set_flip_vertically_on_load(0);
	unsigned char* srcData = stbi_load(sourceFilename, &srcWidth, &srcHeight, &channels, 4);  // Force RGBA

	if (!srcData) {
		std::cerr << "Failed to load foreground for chunking: " << sourceFilename << std::endl;
		std::cerr << "stb_image error: " << stbi_failure_reason() << std::endl;
		return false;
	}

	// Calculate number of tiles in each dimension
	int tilesX = (srcWidth + foreground_chunk_size_width - 1) / foreground_chunk_size_width;   // Ceiling division
	int tilesY = (srcHeight + foreground_chunk_size_height - 1) / foreground_chunk_size_height;

	std::cout << "Chunking foreground (" << srcWidth << "x" << srcHeight << ") into "
		<< tilesX << "x" << tilesY << " tiles of " << foreground_chunk_size_width << "x" << foreground_chunk_size_height << std::endl;

	// Allocate a buffer for each tile (RGBA, 4 bytes per pixel)
	std::vector<unsigned char> tileData(foreground_chunk_size_width * foreground_chunk_size_height * 4, 0);

	// Process each tile
	for (int ty = 0; ty < tilesY; ty++) {
		for (int tx = 0; tx < tilesX; tx++) {
			// Clear the tile buffer (transparent black)
			std::fill(tileData.begin(), tileData.end(), 0);

			// Calculate source region bounds
			int srcStartX = tx * foreground_chunk_size_width;
			int srcStartY = ty * foreground_chunk_size_height;

			// Calculate how many pixels to copy (handle edge tiles that may be smaller)
			int copyWidth = std::min(foreground_chunk_size_width, srcWidth - srcStartX);
			int copyHeight = std::min(foreground_chunk_size_height, srcHeight - srcStartY);

			// Copy pixels from source to tile
			for (int y = 0; y < copyHeight; y++) {
				for (int x = 0; x < copyWidth; x++) {
					int srcIdx = ((srcStartY + y) * srcWidth + (srcStartX + x)) * 4;
					int dstIdx = (y * foreground_chunk_size_width + x) * 4;

					tileData[dstIdx + 0] = srcData[srcIdx + 0];  // R
					tileData[dstIdx + 1] = srcData[srcIdx + 1];  // G
					tileData[dstIdx + 2] = srcData[srcIdx + 2];  // B
					tileData[dstIdx + 3] = srcData[srcIdx + 3];  // A
				}
			}

			// Create OpenGL texture for this tile
			GLuint tileTex;
			glGenTextures(1, &tileTex);
			glBindTexture(GL_TEXTURE_2D, tileTex);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, foreground_chunk_size_width, foreground_chunk_size_height,
				0, GL_RGBA, GL_UNSIGNED_BYTE, tileData.data());

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);






			// Create foreground_tile and add to vector


			bool found_non_transparent = false;

			for (size_t i = 0; i < foreground_chunk_size_width; i++)
			{
				for (size_t j = 0; j < foreground_chunk_size_height; j++)
				{
					size_t index = (j * foreground_chunk_size_width + i) * 4;

					if (tileData[index + 3] > 0)
					{
						found_non_transparent = true;
						i = foreground_chunk_size_width;
						j = foreground_chunk_size_height;
						break;
					}
				}
			}

			if (found_non_transparent == true)
			{
				foreground_tile tile(
					tileTex,
					foreground_chunk_size_width,
					foreground_chunk_size_height,
					static_cast<float>(srcStartX),  // Position in original image coordinates
					static_cast<float>(srcStartY),
					tileData);

				target.push_back(tile);
			}
		}
	}

	for (size_t i = 0; i < target.size(); i++)
	{
		target[i].vel_x = foreground_vel;
		target[i].vel_y = 0;
	}

	// Only the base foreground owns the editor-scroll baseline — skip this for overlays.
	if (&target == &foreground_chunked)
	{
		// Reset the editor-scroll accumulator: any prior editor scroll belonged
		// to the previous level state and has now been discarded along with it.
		g_editorScrollAccum = 0.0f;
		g_loadTimeFgX = target.empty() ? 0.0f : target[0].x;
	}


	stbi_image_free(srcData);

	std::cout << "Created " << target.size() << " foreground tiles" << std::endl;
	return true;
}

/**
 * Draw a texture to the screen using pixel-based coordinates.
 *
 * @param texture     OpenGL texture ID of the sprite to draw
 * @param pixelX      X position of the sprite's top-left corner in window pixels
 * @param pixelY      Y position of the sprite's top-left corner in window pixels
 *                    (0,0) is the top-left corner of the window
 * @param pixelWidth  Width to draw the sprite in window pixels
 * @param pixelHeight Height to draw the sprite in window pixels
 *
 * The coordinate system uses top-left as origin (0,0), with X increasing to the right
 * and Y increasing downward, matching typical screen/image coordinates.
 *
 * Example usage:
 *   // Draw a sprite at position (100, 50) with size 64x64 pixels
 *   drawSprite(myTexture, 100, 50, 64, 64);
 *
 *   // Draw a sprite at the top-left corner of the window
 *   drawSprite(myTexture, 0, 0, spriteWidth, spriteHeight);
 *
 *   // Draw a sprite scaled to 2x its original size
 *   drawSprite(myTexture, 200, 100, originalWidth * 2, originalHeight * 2);
 */
void drawSprite(GLuint texture, int pixelX, int pixelY, int pixelWidth, int pixelHeight, bool under_fire, float alpha) {
	if (texture == 0) return;

	// Enable blending for transparency
	glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glUseProgram(spriteProgram);

	// Convert pixel coordinates to normalized device coordinates [-1, 1]
	// Window coordinate system: (0,0) at top-left, Y increases downward
	// NDC: (-1,-1) at bottom-left, (1,1) at top-right

	// Convert top-left corner position from pixel coords to NDC
	// pixelX=0 -> ndcX=-1, pixelX=windowWidth -> ndcX=1
	// pixelY=0 -> ndcY=1 (top), pixelY=windowHeight -> ndcY=-1 (bottom)
	float ndcX = (2.0f * pixelX / SIM_WIDTH) - 1.0f;
	float ndcY = 1.0f - (2.0f * pixelY / SIM_HEIGHT);  // Flip Y

	// Convert size from pixels to NDC units
	float ndcWidth = 2.0f * pixelWidth / SIM_WIDTH;
	float ndcHeight = 2.0f * pixelHeight / SIM_HEIGHT;

	// The sprite shader expects position as top-left corner in NDC
	// and size as the full width/height in NDC
	// Adjust Y position since we draw from bottom-left of the sprite quad
	float spritePosX = ndcX;
	float spritePosY = ndcY - ndcHeight;  // Move down by height (NDC Y is flipped)

	glUniform2f(glGetUniformLocation(spriteProgram, "spritePos"), spritePosX, spritePosY);
	glUniform2f(glGetUniformLocation(spriteProgram, "spriteSize"), ndcWidth, ndcHeight);

	static const float pi = 4.0f * atanf(1.0f);

	float transformed = (sin(GLOBAL_TIME * 2.0f) + 1.0f) / 2.0f;

	if (alpha == 1.0)
		transformed = alpha;

	glUniform1f(glGetUniformLocation(spriteProgram, "alpha"), transformed);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glUniform1i(glGetUniformLocation(spriteProgram, "spriteTexture"), 0);

	glUniform1i(glGetUniformLocation(spriteProgram, "under_fire"), under_fire);
	glUniform1f(glGetUniformLocation(spriteProgram, "time"), GLOBAL_TIME);


	drawQuad();

	glDisable(GL_BLEND);
}


/**
 * Draw a health bar above a sprite.
 *
 * @param pixelX        X position of the sprite's top-left corner in window pixels
 * @param pixelY        Y position of the sprite's top-left corner in window pixels
 * @param spriteWidth   Width of the sprite in pixels
 * @param health        Current health value
 * @param maxHealth     Maximum health value
 * @param barWidth      Width of the health bar in pixels (default: sprite width)
 * @param barHeight     Height of the health bar in pixels (default: 8)
 * @param yOffset       Vertical offset above the sprite in pixels (default: 10)
 */
void drawHealthBar(int pixelX, int pixelY, int spriteWidth, float health, float maxHealth,
	int barWidth = -1, int barHeight = 8, int yOffset = 10)
{
	if (maxHealth <= 0) return;

	// Use sprite width as default bar width
	if (barWidth < 0) barWidth = spriteWidth;

	// Calculate health percentage (clamped to 0-1)
	float healthPercent = std::max(0.0f, std::min(1.0f, health / maxHealth));

	// Position the bar above the sprite
	int barX = pixelX + (spriteWidth - barWidth) / 2;  // Center the bar over sprite
	int barY = pixelY - yOffset - barHeight;           // Position above sprite

	// Enable blending for transparency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glUseProgram(healthBarProgram);

	// Convert pixel coordinates to normalized device coordinates [-1, 1]
	auto pixelToNDC = [](int px, int py, int pw, int ph) {
		float ndcX = (2.0f * px / SIM_WIDTH) - 1.0f;
		float ndcY = 1.0f - (2.0f * py / SIM_HEIGHT);  // Flip Y
		float ndcWidth = 2.0f * pw / SIM_WIDTH;
		float ndcHeight = 2.0f * ph / SIM_HEIGHT;

		// Adjust Y position since we draw from bottom-left
		float barPosX = ndcX;
		float barPosY = ndcY - ndcHeight;

		return std::make_tuple(barPosX, barPosY, ndcWidth, ndcHeight);
		};

	// Draw background bar (dark gray)
	{
		auto [posX, posY, width, height] = pixelToNDC(barX, barY, barWidth, barHeight);
		glUniform2f(glGetUniformLocation(healthBarProgram, "barPos"), posX, posY);
		glUniform2f(glGetUniformLocation(healthBarProgram, "barSize"), width, height);
		glUniform4f(glGetUniformLocation(healthBarProgram, "barColor"), 0.2f, 0.2f, 0.2f, 0.25f);
		drawQuad();
	}

	// Draw foreground bar (colored based on health)
	if (healthPercent > 0.0f)
	{
		int fillWidth = static_cast<int>(barWidth * healthPercent);
		auto [posX, posY, width, height] = pixelToNDC(barX, barY, fillWidth, barHeight);

		// Color gradient: green -> yellow -> red based on health
		float r, g, b;
		if (healthPercent > 0.5f) {
			// Green to yellow (health 100% to 50%)
			float t = (healthPercent - 0.5f) * 2.0f;
			r = 1.0f - t;
			g = 1.0f;
			b = 0.0f;
		}
		else {
			// Yellow to red (health 50% to 0%)
			float t = healthPercent * 2.0f;
			r = 1.0f;
			g = t;
			b = 0.0f;
		}

		glUniform2f(glGetUniformLocation(healthBarProgram, "barPos"), posX, posY);
		glUniform2f(glGetUniformLocation(healthBarProgram, "barSize"), width, height);
		glUniform4f(glGetUniformLocation(healthBarProgram, "barColor"), r, g, b, 0.75f);
		drawQuad();
	}

	//// Draw border (black outline)
	//{
	//	auto [posX, posY, width, height] = pixelToNDC(barX - 1, barY - 1, barWidth + 2, barHeight + 2);
	//	glUniform2f(glGetUniformLocation(healthBarProgram, "barPos"), posX, posY);
	//	glUniform2f(glGetUniformLocation(healthBarProgram, "barSize"), width, height);
	//	glUniform4f(glGetUniformLocation(healthBarProgram, "barColor"), 0.0f, 0.0f, 0.0f, 1.0f);

	//	// Draw only the outline by drawing 4 thin rectangles
	//	// This is a simple approach - for a proper outline you'd use a different technique
	//	// For simplicity, we'll skip the outline and just use the filled bars above
	//}

	glDisable(GL_BLEND);
}


void fireBullet(void)
{
	if (protagonist.to_be_culled)
		return;

	std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> timeSinceLastBullet = currentTime - lastBulletTime;

	if (timeSinceLastBullet.count() < PROTAGONIST_MIN_BULLET_INTERVAL)
		return;

	lastBulletTime = currentTime;



	if (sinusoidal_fire)
	{
		sine_bullet s;

		s.tex = bullet_template.tex;
		s.to_present_data = bullet_template.to_present_data;

		// Position in PIXELS (no normalization)
		s.x = protagonist.x + protagonist.width;
		s.y = protagonist.y + protagonist.height / 2.0f;
		s.width = bullet_template.width;
		s.height = bullet_template.height;

		static const float pi = 4.0f * atanf(1.0f);

		float angle_start = 0, angle_end = 0;
		size_t num_streams = 1;

		if (x3_fire) { angle_start = 0.4f; angle_end = -0.4f; num_streams = 3; }
		if (x5_fire) { angle_start = 0.4f; angle_end = -0.4f; num_streams = 5; }

		float angle_step = (num_streams > 1) ? (angle_end - angle_start) / (num_streams - 1) : 0;
		float angle = angle_start;

		// Bullet speed in PIXELS per second
		const float BULLET_SPEED = 1600.0;  // Adjust as needed

		for (size_t i = 0; i < num_streams; i++, angle += angle_step)
		{
			sine_bullet newBullet = s;
			newBullet.vel_x = BULLET_SPEED * cos(angle);  // pixels/sec
			newBullet.vel_y = BULLET_SPEED * sin(angle);  // pixels/sec
			newBullet.sinusoidal_shift = false;
			newBullet.sinusoidal_amplitude = 600 / DT;  // amplitude in PIXELS
			newBullet.sinusoidal_frequency = 10;
			newBullet.birth_time = GLOBAL_TIME;
			newBullet.death_time = -1;

			ally_bullets.push_back(make_unique<sine_bullet>(newBullet));

			newBullet.sinusoidal_shift = true;
			ally_bullets.push_back(make_unique<sine_bullet>(newBullet));
		}
	}
	else
	{
		straight_bullet s;

		s.tex = bullet_template.tex;
		s.to_present_data = bullet_template.to_present_data;

		// Position in PIXELS (no normalization)
		s.x = protagonist.x + protagonist.width;
		s.y = protagonist.y + protagonist.height / 2.0f;
		s.width = bullet_template.width;
		s.height = bullet_template.height;

		static const float pi = 4.0f * atanf(1.0f);

		float angle_start = 0, angle_end = 0;
		size_t num_streams = 1;

		if (x3_fire) { angle_start = 0.4f; angle_end = -0.4f; num_streams = 3; }
		if (x5_fire) { angle_start = 0.4f; angle_end = -0.4f; num_streams = 5; }

		float angle_step = (num_streams > 1) ? (angle_end - angle_start) / (num_streams - 1) : 0;
		float angle = angle_start;

		// Bullet speed in PIXELS per second
		const float BULLET_SPEED = 1600.0;  // Adjust as needed

		for (size_t i = 0; i < num_streams; i++, angle += angle_step)
		{
			straight_bullet newBullet = s;
			newBullet.vel_x = BULLET_SPEED * cos(angle);  // pixels/sec
			newBullet.vel_y = BULLET_SPEED * sin(angle);  // pixels/sec
			newBullet.birth_time = GLOBAL_TIME;
			newBullet.death_time = -1;

			ally_bullets.push_back(make_unique<straight_bullet>(newBullet));
		}
	}

	laser_sound.play();
}




void make_dying_bullets(const pre_sprite& stamp, const bool enemy)
{
	if (stamp.to_be_culled)
		return;


	explosion_sound.play();


	//const float aspect = SIM_WIDTH / float(SIM_HEIGHT);

	//std::chrono::high_resolution_clock::time_point global_time_end = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<float, std::milli> elapsed;
	//elapsed = global_time_end - app_start_time;

	bullet newCentralStamp = bullet_template;

	float x_rad = 0.5f * stamp.width / float(SIM_WIDTH);
	//float y_rad = 0.5f * stamp.height / float(SIM_HEIGHT);

	float avg_rad = 0.1f * x_rad;

	newCentralStamp.x = stamp.x + stamp.width / 2.0f;
	newCentralStamp.y = stamp.y + stamp.height / 2.0f;

	newCentralStamp.birth_time = GLOBAL_TIME;
	newCentralStamp.death_time = GLOBAL_TIME + 0.1f;

	newCentralStamp.density_add = avg_rad;
	newCentralStamp.velocity_add = avg_rad;



	//	newCentralStamp.is_dying_bullet = true;

	if (enemy)
		enemy_bullets.push_back(make_unique<straight_bullet>(newCentralStamp));
	else
		ally_bullets.push_back(make_unique<straight_bullet>(newCentralStamp));

	return;



	//for (size_t j = 0; j < 3; j++)
	//{
	//	bullet newStamp = newCentralStamp;

	//	newStamp.density_add = avg_rad / 4;
	//	newStamp.velocity_add = avg_rad / 4;


	//	RandomUnitVector(newStamp.vel_x, newStamp.vel_y);

	//	newStamp.vel_x *= aspect * 100 * (rand() / float(RAND_MAX));
	//	newStamp.vel_y *= 100 * (rand() / float(RAND_MAX));

	//	//		newStamp.path_randomization = (rand() / float(RAND_MAX)) * 0.01f;

	//	newStamp.birth_time = GLOBAL_TIME;
	//	newStamp.death_time = GLOBAL_TIME + 1.0f * rand() / float(RAND_MAX);

	//	if (enemy)
	//		enemy_bullets.push_back(make_unique<straight_bullet>(newStamp));
	//	else
	//		ally_bullets.push_back(make_unique<straight_bullet>(newStamp));
	//}

	//for (size_t j = 0; j < 5; j++)
	//{
	//	bullet newStamp = newCentralStamp;

	//	newStamp.density_add = avg_rad / 8;
	//	newStamp.velocity_add = avg_rad / 8;

	//	RandomUnitVector(newStamp.vel_x, newStamp.vel_y);

	//	newStamp.vel_x *= aspect * 100 * (rand() / float(RAND_MAX));
	//	newStamp.vel_y *= 100 * (rand() / float(RAND_MAX));
	//	//rnewStamp.path_randomization = (rand() / float(RAND_MAX)) * 0.01f;
	//	newStamp.birth_time = GLOBAL_TIME;
	//	newStamp.death_time = GLOBAL_TIME + 3.0f * rand() / float(RAND_MAX);

	//	if (enemy)
	//		enemy_bullets.push_back(make_unique<straight_bullet>(newStamp));
	//	else
	//		ally_bullets.push_back(make_unique<straight_bullet>(newStamp));
	//}
}

// Returns cannon local (x, y) remapped from the rest state to the current state.
// Cannon positions are defined in editor mode (rest state); this maps them so
// they track the ship's visual body when the sprite tilts.
static glm::vec2 getCannonLocalPos(const enemy_ship& e, const cannon& c)
{
	const int rest = e.rest_state_index();
	const int cur = e.state;

	if (cur == rest || e.num_frames() <= 1 ||
		cur < 0 || cur >= (int)e.original_sprite_frames.size() ||
		rest < 0 || rest >= (int)e.original_sprite_frames.size())
		return glm::vec2((float)c.x, (float)c.y);

	const unsigned char* rest_data = e.original_sprite_frames[rest].data();
	const unsigned char* cur_data = e.original_sprite_frames[cur].data();
	if (!rest_data || !cur_data)
		return glm::vec2((float)c.x, (float)c.y);


	int col = std::max(0, std::min((int)(c.x + 0.5), e.width - 1));

	// Find vertical extent of visible pixels in the rest state at this column
	int src_first = -1, src_last = -1;
	for (int row = 0; row < e.height; ++row)
	{
		if (rest_data[(row * e.width + col) * 4 + 3] > 0)
		{
			if (src_first == -1) src_first = row;
			src_last = row;
		}
	}
	if (src_first == -1 || src_last == src_first)
		return glm::vec2((float)c.x, (float)c.y);  // degenerate, can't remap

	// Normalize cannon.y within the rest-state extent
	float t_norm = ((float)c.y - (float)src_first) / (float)(src_last - src_first);

	t_norm = std::max(0.0f, std::min(1.0f, t_norm));  // clamp!

	// Find vertical extent in the current state at the same column
	int dst_first = -1, dst_last = -1;
	for (int row = 0; row < e.height; ++row)
	{
		if (cur_data[(row * e.width + col) * 4 + 3] > 0)
		{
			if (dst_first == -1) dst_first = row;
			dst_last = row;
		}
	}
	if (dst_first == -1 || dst_last == dst_first)
		return glm::vec2((float)c.x, (float)c.y);

	// Cosine-based mapping: rotation compresses/expands non-linearly
	//float cos_t = 0.5f * (1.0f - cosf(t_norm * 3.14159265f));

	//	float mapped_y = (float)dst_first + cos_t * (float)(dst_last - dst_first);
	float mapped_y = (float)dst_first + t_norm * (float)(dst_last - dst_first);
	return glm::vec2((float)c.x, mapped_y);
}




void simulate()
{
	if (spacePressed)
		fireBullet();



	protagonist.updateTiltFromInput();
	protagonist.applyInertia(DT);

	protagonist.integrate(DT);


	protagonist.x = std::max(0.0f, std::min(protagonist.x, (float)(SIM_WIDTH - protagonist.width)));
	protagonist.y = std::max(0.0f, std::min(protagonist.y, (float)(SIM_HEIGHT - protagonist.height)));






	for (size_t i = 0; i < enemy_ships.size(); i++)
	{
		enemy_ships[i]->updateTiltFromInput();

		if (enemy_ships[i]->to_be_culled || false == enemy_ships[i]->isOnscreen())
			continue;

		for (size_t j = 0; j < enemy_ships[i]->cannons.size(); j++)
		{

			/*			double x = enemy_ships[i]->cannons[j].x;
						double y = enemy_ships[i]->cannons[j].y;
					*/
			glm::vec2 local = getCannonLocalPos(*enemy_ships[i], enemy_ships[i]->cannons[j]);
			double x = local.x;
			double y = local.y;

			// Skip firing if the cannon location is transparent in the sprite
			{
				int px = static_cast<int>(x);
				int py = static_cast<int>(y);

				bool transparent = false;

				if (px < 0 || px >= enemy_ships[i]->width ||
					py < 0 || py >= enemy_ships[i]->height)
				{
					transparent = true;
				}
				else
				{
					// Use the current active frame data for transparency check
					const unsigned char* frame_data = getTriSpriteActiveData(*enemy_ships[i]);
					if (frame_data)
					{
						size_t index = (static_cast<size_t>(py) * enemy_ships[i]->width + px) * 4 + 3;
						transparent = (frame_data[index] == 0);
					}
				}

				if (transparent)
				{
					//cout << "skipping cannon " << j << endl;
					continue;
				}
			}



			std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> timeSinceLastBullet = currentTime - enemy_ships[i]->cannons[j].lastBulletTime;

			if (timeSinceLastBullet.count() < enemy_ships[i]->cannons[j].min_bullet_interval)
				continue;

			enemy_ships[i]->cannons[j].lastBulletTime = currentTime;



			straight_bullet s;
			s.tex = bullet_template.tex;
			s.to_present_data = bullet_template.to_present_data;
			s.width = bullet_template.width;
			s.height = bullet_template.height;

			//s.x = enemy_ships[i]->x + enemy_ships[i]->cannons[j].x;
			//s.y = enemy_ships[i]->y + enemy_ships[i]->cannons[j].y;
			s.x = enemy_ships[i]->x + local.x;   // was cannons[j].x
			s.y = enemy_ships[i]->y + local.y;   // was cannons[j].y

			const float BULLET_SPEED = 1600.0;  // Adjust as needed

			if (enemy_ships[i]->cannons[j].cannon_type == CANNON_TYPE_LEFT)
			{
				s.vel_x = -BULLET_SPEED;
				enemy_bullets.push_back(make_unique<straight_bullet>(s));
			}
			else if (enemy_ships[i]->cannons[j].cannon_type == CANNON_TYPE_UP_DOWN)
			{
				s.vel_y = -BULLET_SPEED;
				enemy_bullets.push_back(make_unique<straight_bullet>(s));

				s.vel_y = BULLET_SPEED;
				enemy_bullets.push_back(make_unique<straight_bullet>(s));
			}
			else if (enemy_ships[i]->cannons[j].cannon_type == CANNON_TYPE_TRACKING)
			{
				glm::vec2 aim;
				aim.x = (protagonist.x + protagonist.width * 0.5f) - s.x;
				aim.y = (protagonist.y + protagonist.height * 0.5f) - s.y;

				aim = normalize(aim) * BULLET_SPEED;

				s.vel_x = aim.x;
				s.vel_y = aim.y;

				enemy_bullets.push_back(make_unique<straight_bullet>(s));
			}
			else if (enemy_ships[i]->cannons[j].cannon_type == CANNON_TYPE_CIRCULAR)
			{
				const int NUM_BULLETS = 12; // adjust for denser/sparser spread
				const float TWO_PI = 2.0f * (4.0f * atanf(1.0f));

				for (int k = 0; k < NUM_BULLETS; k++)
				{
					float angle = (TWO_PI / NUM_BULLETS) * k;

					straight_bullet sb;
					sb.tex = s.tex;
					sb.to_present_data = s.to_present_data;
					sb.width = s.width;
					sb.height = s.height;
					sb.x = s.x;
					sb.y = s.y;
					sb.vel_x = BULLET_SPEED * cos(angle);
					sb.vel_y = BULLET_SPEED * sin(angle);

					enemy_bullets.push_back(make_unique<straight_bullet>(sb));
				}
			}
			//cout << "shoot" << endl;
		}

	}










	for (size_t i = 0; i < enemy_ships.size(); i++)
	{
		// Activate path when enemy drifts onscreen (for level-loaded enemies with path_t == -1)
		if (enemy_ships[i]->isOnscreen() && enemy_ships[i]->path_t == -1)
		{
			enemy_ships[i]->path_t = 0.0;
			// The path knots are already placed at their correct screen-space
			// positions (knot 0 at SIM_WIDTH + half_w, last knot at -half_w).
			// No scrolling during the spline phase: the rightmost knot stays at
			// SIM_WIDTH + half_w while the enemy traverses to the leftmost knot
			// at -half_w.
			enemy_ships[i]->path_scroll_rate = 0.0f;
		}

		// Choose scroll rate depending on phase:
		//   pre-activation  -> foreground_vel (keep path aligned with drifting enemy)
		//   active spline   -> 0 (path knots are fixed in screen space)
		//   post-animation  -> foreground_vel (resume world drift)
		float scroll_rate;
		if (enemy_ships[i]->path_t >= 0.0f && enemy_ships[i]->path_t <= 1.0f)
			scroll_rate = enemy_ships[i]->path_scroll_rate;
		else
			scroll_rate = foreground_vel;

		for (size_t j = 0; j < enemy_ships[i]->path_points.size(); j++)
			enemy_ships[i]->path_points[j].x += scroll_rate * DT;

		// While the enemy is on the spline, path_points are frozen in
		// screen space but the foreground keeps scrolling. Track the
		// resulting drift so editorSaveToDatabase can add it back when
		// canonicalising path_points (otherwise the saved x would be
		// over-corrected by exactly this amount, sliding the enemy
		// rightward across save/load cycles).
		if (enemy_ships[i]->path_t >= 0.0f && enemy_ships[i]->path_t <= 1.0f)
			enemy_ships[i]->spline_phase_drift += foreground_vel * DT;

		if (enemy_ships[i]->isOnscreen() && enemy_ships[i]->appearance_time == 0)
			enemy_ships[i]->appearance_time = GLOBAL_TIME;

		// Spline-driven movement: always follow path when path_t is active,
		// even if the enemy is currently offscreen (e.g. starting from an offscreen knot)
		if (enemy_ships[i]->path_t >= 0.0f && enemy_ships[i]->path_t <= 1.0f)
		{
			float t = enemy_ships[i]->path_t;

			// Get speed multiplier at current path position
			float speed_mult = get_spline_point(enemy_ships[i]->path_speeds, t);

			// Advance path_t based on speed (base rate = 1/path_animation_length per second)
			float base_rate = 1.0f / enemy_ships[i]->path_animation_length;
			enemy_ships[i]->path_t += base_rate * speed_mult * DT;

			// Set position directly from spline (more stable than velocity integration)
			glm::vec2 pos = get_spline_point(enemy_ships[i]->path_points, t);
			enemy_ships[i]->old_x = enemy_ships[i]->x;
			enemy_ships[i]->old_y = enemy_ships[i]->y;
			enemy_ships[i]->x = pos.x - enemy_ships[i]->width * 0.5f;
			enemy_ships[i]->y = pos.y - enemy_ships[i]->height * 0.5f;


			//enemy_ships[i]->vel_x = 0;
			//enemy_ships[i]->vel_y = 0;


			glm::vec2 tangent = get_spline_tangent(enemy_ships[i]->path_points, t);
			float num_segments = (float)(enemy_ships[i]->path_points.size() - 1);
			float speed_scale = num_segments / enemy_ships[i]->path_animation_length;
			enemy_ships[i]->vel_x = tangent.x * speed_mult * speed_scale;
			enemy_ships[i]->vel_y = tangent.y * speed_mult * speed_scale;

			enemy_ships[i]->set_velocity(enemy_ships[i]->vel_x, enemy_ships[i]->vel_y);
		}
		else if (enemy_ships[i]->isOnscreen())
		{
			enemy_ships[i]->vel_x = foreground_vel;
			enemy_ships[i]->vel_y = 0;

			enemy_ships[i]->set_velocity(enemy_ships[i]->vel_x, enemy_ships[i]->vel_y);
			enemy_ships[i]->integrate(DT);
		}
		else
		{
			if (enemy_ships[i]->x < 0)
			{
				if (enemy_ships[i]->to_be_culled == false)
					cout << enemy_ships[i]->x << endl;

				enemy_ships[i]->to_be_culled = true;
			}
			else
			{
				enemy_ships[i]->vel_x = foreground_vel;
				enemy_ships[i]->vel_y = 0;

				enemy_ships[i]->set_velocity(enemy_ships[i]->vel_x, enemy_ships[i]->vel_y);
				enemy_ships[i]->integrate(DT);
			}


		}


	}



	// First, integrate ALL foreground chunks (movement must happen regardless of collision)
	for (size_t i = 0; i < foreground_chunked.size(); i++)
	{
		foreground_chunked[i].integrate(DT);
	}

	// Integrate the lit overlay so it scrolls in lockstep with the base foreground.
	// Purely visual — not part of collision resolution.
	for (size_t i = 0; i < foreground_lit_chunked.size(); i++)
	{
		foreground_lit_chunked[i].integrate(DT);
	}

	// Calculate how much the foreground moved this frame
	// This is needed to properly resolve collisions with moving geometry
	const float foreground_dx = foreground_vel * DT;

	// Then, check for collision with protagonist separately
	bool resolved = false;

	for (size_t i = 0; i < foreground_chunked.size() && !resolved; i++)
	{
		if (false == foreground_chunked[i].isOnscreen())
			continue;

		if (detectTriSpriteToSpriteOverlap(protagonist, foreground_chunked[i], 1))
		{
			const float DAMAGE_INTERVAL = aberrationDuration;  // seconds between damage ticks

			if (GLOBAL_TIME - protagonist.last_time_collided >= DAMAGE_INTERVAL)
			{
				protagonist.health -= 100.0f;

				protagonist.last_time_collided = GLOBAL_TIME;

				// Trigger chromatic aberration effect on collision damage
				lastDamageTime = GLOBAL_TIME;
			}


			// Test X resolution - account for foreground movement
			// The protagonist's old_x was valid when foreground was at ITS old position,
			// so we must also shift by how much the foreground moved
			float tempX = protagonist.x;
			protagonist.x = protagonist.old_x + foreground_dx;

			bool xResolves = true;
			// Check if moving only X back (plus foreground offset) resolves ALL collisions
			for (size_t j = 0; j < foreground_chunked.size(); j++)
			{
				if (foreground_chunked[j].isOnscreen() &&
					detectTriSpriteToSpriteOverlap(protagonist, foreground_chunked[j], 1))
				{
					xResolves = false;
					break;
				}
			}

			if (xResolves)
			{
				protagonist.vel_x = 0;  // Block horizontal movement
				resolved = true;

				// Kill protagonist if pushed offscreen by foreground
				if (!protagonist.isOnscreen())
				{
					protagonist.health = 0;
				}

				continue;
			}

			// Test Y resolution
			protagonist.x = tempX;  // restore X
			protagonist.y = protagonist.old_y;

			bool yResolves = true;
			for (size_t j = 0; j < foreground_chunked.size(); j++)
			{
				if (foreground_chunked[j].isOnscreen() &&
					detectTriSpriteToSpriteOverlap(protagonist, foreground_chunked[j], 1))
				{
					yResolves = false;
					break;
				}
			}

			if (yResolves)
			{
				protagonist.vel_y = 0;  // Block vertical movement
				resolved = true;

				// Kill protagonist if pushed offscreen by foreground
				if (!protagonist.isOnscreen())
				{
					protagonist.health = 0;
				}

				continue;
			}

			// Neither alone works -> full revert (corner case)
			// Apply foreground offset to X to maintain separation
			protagonist.x = protagonist.old_x;// +foreground_dx;
			protagonist.y = protagonist.old_y;
			protagonist.vel_x = 0;
			protagonist.vel_y = 0;
			resolved = true;

			// If still colliding after full revert, protagonist is being crushed
			// Check and apply additional pushback or damage
			bool stillColliding = false;
			for (size_t j = 0; j < foreground_chunked.size(); j++)
			{
				if (foreground_chunked[j].isOnscreen() &&
					detectTriSpriteToSpriteOverlap(protagonist, foreground_chunked[j], 1))
				{
					stillColliding = true;
					break;
				}
			}

			if (stillColliding)
			{
				// Protagonist is being crushed - push them along with foreground
				// or apply continuous damage. Here we push them out.
				protagonist.x += foreground_dx;

				// Kill protagonist if pushed offscreen by foreground
				if (!protagonist.isOnscreen())
				{
					protagonist.health = 0;
				}
			}
		}
	}




	for (size_t i = 0; i < enemy_ships.size(); i++)
	{
		if (false == enemy_ships[i]->isOnscreen() || true == enemy_ships[i]->to_be_culled)
			continue;

		const float DAMAGE_INTERVAL = aberrationDuration;

		if (detectTriSpriteOverlap(protagonist, *enemy_ships[i], 1) &&
			GLOBAL_TIME - protagonist.last_time_collided >= DAMAGE_INTERVAL)
		{
			protagonist.health -= 100.0f;
			enemy_ships[i]->health -= 100.0f;
			lastDamageTime = GLOBAL_TIME;

			protagonist.last_time_collided = GLOBAL_TIME;
		}
	}



	for (auto it = ally_bullets.begin(); it != ally_bullets.end(); it++)
	{
		(*it)->integrate(DT);

		bool found_collision = false;

		for (size_t i = 0; i < foreground_chunked.size(); i++)
		{
			if (false == foreground_chunked[i].isOnscreen())
				continue;

			found_collision = detectSpriteOverlap(*(*it), foreground_chunked[i], 1);

			if (true == found_collision)
				break;
		}

		if (false == found_collision)
		{
			for (size_t i = 0; i < enemy_ships.size(); i++)
			{
				if (false == enemy_ships[i]->isOnscreen() || enemy_ships[i]->to_be_culled)
					continue;

				found_collision = detectTriSpriteToSpriteOverlap(*enemy_ships[i], *(*it), 1);

				if (true == found_collision)
					break;
			}
		}

		if (false == (*it)->isOnscreen() || found_collision || ((*it)->death_time != -1 && (*it)->death_time <= GLOBAL_TIME))
		{
			(*it)->to_be_culled = true;
		}
	}










	for (auto it = enemy_bullets.begin(); it != enemy_bullets.end(); it++)
	{
		if ((*it)->to_be_culled)
			continue;

		(*it)->integrate(DT);

		bool found_collision = false;

		for (size_t i = 0; i < foreground_chunked.size(); i++)
		{
			if (false == foreground_chunked[i].isOnscreen())
				continue;

			found_collision = detectSpriteOverlap(*(*it), foreground_chunked[i], 1);

			if (true == found_collision)
			{
				cout << "ENEMY BULLET - FOREGROUND COLLISION" << endl;

				(*it)->to_be_culled = true;
				break;
			}
		}

		if (false == found_collision)
		{
			found_collision = detectTriSpriteToSpriteOverlap(protagonist, *(*it), 1);

			if (true == found_collision)
			{
				(*it)->to_be_culled = true;  // Mark bullet for removal so it doesn't keep triggering
			}
		}


		if (false == (*it)->isOnscreen() || found_collision || ((*it)->death_time != -1 && (*it)->death_time <= GLOBAL_TIME))
		{
			(*it)->to_be_culled = true;
		}

	}





	// Integrate live power-ups and flag off-screen ones for culling
	for (auto& p : power_ups_alive)
	{
		if (p->to_be_culled)
			continue;

		p->integrate(DT);

		if (!p->isOnscreen())
			p->to_be_culled = true;

		if (detectTriSpriteToSpriteOverlap(protagonist, *p, 1))
		{
			if (dynamic_cast<sinusoidal_fire_power_up*>(p.get()))
			{
				sinusoidal_fire = true;
			}
			else if (dynamic_cast<x3_fire_power_up*>(p.get()))
			{
				x3_fire = true;
			}
			else if (dynamic_cast<x5_fire_power_up*>(p.get()))
			{
				x5_fire = true;
			}

			p->to_be_culled = true;
		}

	}



	// NEW: Build batched splats
	std::vector<Splat> densitySplats;
	std::vector<Splat> velocitySplats;

	auto accumulateBulletSplats = [&](const std::vector<std::unique_ptr<bullet>>& bullets, bool isRedChannel) {
		for (size_t i = 0; i < bullets.size(); ++i) {
			auto& b = bullets[i];

			int pathSamples = 10; // keep your sampling density
			float prevX = b->old_x;
			float prevY = b->old_y;

			for (int step = 0; step <= pathSamples; ++step) {
				float t = (float)step / pathSamples;
				float sampleX = prevX + (b->x - prevX) * t;
				float sampleY = prevY + (b->y - prevY) * t;

				float normX = pixelToNormX(sampleX);
				float normY = pixelToNormY(sampleY);

				// Density splat: RG channels; put into vx,vy (vz used for third component which display shader ignores)
				Splat ds;
				ds.px = normX;
				ds.py = normY;
				if (isRedChannel) {
					ds.vx = 1.0f; // red density -> .x
					ds.vy = 0.0f; // green density -> .y
				}
				else {
					ds.vx = 0.0f;
					ds.vy = 1.0f;
				}
				ds.vz = 0.0f;
				ds.radius = b->density_add;

				densitySplats.push_back(ds);

				// Velocity splat: add actual per-frame velocity (already normalized by your helpers)
				float actualVelX = (b->x - b->old_x) / DT;
				float actualVelY = (b->y - b->old_y) / DT;
				float normVelX = 0.1f * velPixelToNormX(actualVelX);
				float normVelY = 0.1f * velPixelToNormY(actualVelY);

				Splat vs;
				vs.px = normX;
				vs.py = normY;
				vs.vx = normVelX;
				vs.vy = normVelY;
				vs.vz = 0.0f;
				vs.radius = b->velocity_add;

				velocitySplats.push_back(vs);
			}
		}
		};

	// Ally bullets add red density
	accumulateBulletSplats(ally_bullets, /*isRedChannel=*/true);
	// Enemy bullets add green density
	accumulateBulletSplats(enemy_bullets, /*isRedChannel=*/false);

	// NEW: Submit batched splats (density and velocity separately)
	addSourcesBatch(densityTex, densityFBO, currentDensity, densitySplats);
	addSourcesBatch(velocityTex, velocityFBO, currentVelocity, velocitySplats);




	if (protagonist.health <= 0)
	{
		// Trigger final damage effect
		lastDamageTime = GLOBAL_TIME;

		make_dying_bullets(protagonist, false);
		protagonist.to_be_culled = true;
	}

	for (size_t i = 0; i < enemy_ships.size(); ++i)
	{
		if (enemy_ships[i]->to_be_culled)
			continue;

		if (enemy_ships[i]->health <= 0)
		{
			// ============== TRIGGER WAVE EFFECT ON ENEMY DEATH ==============
			// Trigger wave chromatic aberration at enemy's death position
			triggerWaveAtPosition(
				enemy_ships[i]->x,
				enemy_ships[i]->y,
				static_cast<float>(enemy_ships[i]->width),
				static_cast<float>(enemy_ships[i]->height)
			);
			// ============== END WAVE TRIGGER ==============

			make_dying_bullets(*(enemy_ships[i]), true);

			// spawn power ups here
			for (int pu_type : enemy_ships[i]->power_ups)
			{
				sprite* tmpl = nullptr;
				if (pu_type == POWER_UP_TYPE_SINUSOIDAL) tmpl = &power_up_template_sinusoidal;
				else if (pu_type == POWER_UP_TYPE_X3)    tmpl = &power_up_template_x3;
				else if (pu_type == POWER_UP_TYPE_X5)    tmpl = &power_up_template_x5;
				if (!tmpl || tmpl->tex == 0) continue;

				std::unique_ptr<power_up> p;
				if (pu_type == POWER_UP_TYPE_SINUSOIDAL)
					p = make_unique<sinusoidal_fire_power_up>();
				else if (pu_type == POWER_UP_TYPE_X3)
					p = make_unique<x3_fire_power_up>();
				else if (pu_type == POWER_UP_TYPE_X5)
					p = make_unique<x5_fire_power_up>();
				else
					p = make_unique<power_up>();
				p->tex = tmpl->tex;
				p->width = tmpl->width;
				p->height = tmpl->height;
				p->to_present_data = tmpl->to_present_data;
				p->to_present_data_pointers.clear();
				if (!p->to_present_data.empty())
					p->to_present_data_pointers.push_back(p->to_present_data.data());
				// Center on the dying enemy
				p->x = enemy_ships[i]->x + (enemy_ships[i]->width - p->width) / 2.0f;
				p->y = enemy_ships[i]->y + (enemy_ships[i]->height - p->height) / 2.0f;
				// Drift left with the world so it appears stationary in world space
				p->vel_x = foreground_vel;
				p->vel_y = 0.0f;
				power_ups_alive.push_back(std::move(p));
			}

			enemy_ships[i]->to_be_culled = true;
		}
	}













	GLuint clearColor[4] = { 0, 0, 0, 0 };
	glClearTexImage(obstacleTex, 0, GL_RGBA, GL_UNSIGNED_BYTE, clearColor);



	if (protagonist.to_be_culled == false)
	{
		addObstacleStamp(protagonist.tex,
			static_cast<int>(protagonist.x), static_cast<int>(protagonist.y),
			protagonist.width, protagonist.height, true,
			1, true);
	}

	for (size_t i = 0; i < foreground_chunked.size(); i++)
	{
		if (foreground_chunked[i].tex != 0 && foreground_chunked[i].isOnscreen())
		{
			addObstacleStamp(foreground_chunked[i].tex,
				static_cast<int>(foreground_chunked[i].x), static_cast<int>(foreground_chunked[i].y),
				foreground_chunked[i].width, foreground_chunked[i].height, true,
				0.5, true);
		}
	}

	for (size_t i = 0; i < enemy_ships.size(); i++)
	{
		if (enemy_ships[i]->tex != 0 && enemy_ships[i]->isOnscreen() && false == enemy_ships[i]->to_be_culled)
		{
			addObstacleStamp(enemy_ships[i]->tex,
				static_cast<int>(enemy_ships[i]->x), static_cast<int>(enemy_ships[i]->y),
				enemy_ships[i]->width, enemy_ships[i]->height, true,
				0.5, true);
		}
	}



	// Advect velocity
	advect(velocityTex[currentVelocity], velocityTex[currentVelocity], velocityFBO[1 - currentVelocity], VELOCITY_DISSIPATION);
	currentVelocity = 1 - currentVelocity;

	// Advect density
	advect(velocityTex[currentVelocity], densityTex[currentDensity], densityFBO[1 - currentDensity], DENSITY_DISSIPATION);
	currentDensity = 1 - currentDensity;

	// Vorticity confinement
	if (1/*vorticityEnabled*/) {
		computeVorticity();
		applyVorticityForce();
	}

	//applyTurbulence();

	applyTurbulence();

	// Pressure projection
	computeDivergence();
	clearPressure();
	jacobi();
	subtractGradient();




}













// =============================================================================
//  EDITOR MODE
//
//  CONTROLS (Tab to toggle)
//  ──────────────────────────────────────
//  Tab               Toggle editor on/off
//
//  Left / Right      Scroll foreground (and enemies) back / forth
//
//  [ / ]             Select previous / next enemy ship
//  N                 Spawn new enemy (cycles through loaded templates)
//  Delete            Remove selected enemy
//
//  LMB near knot     Select & drag that Catmull-Rom control point
//  LMB on empty      Insert new control point in nearest segment
//  RMB near knot     Delete that control point (minimum 2 kept)
//
//  C                 Add a cannon at current mouse position (sprite-local coords)
//  V                 Remove the selected cannon (or last if none selected)
//  { / }             Select previous / next cannon on the selected enemy
//  T                 Cycle cannon type of selected cannon: LEFT -> UP_DOWN -> TRACKING
//  , / .             Decrease / increase fire interval of selected cannon
//
//  h / H             Decrease / increase selected enemy max_health by 100
//  j / J             Decrease / increase selected enemy path_animation_length by 1.0s
//
//  Shift+LMB         Add a speed knot (default value 1.0)
//  Shift+RMB         Remove last speed knot
//  LMB near diamond  Select that speed knot (shown in yellow)
//  Scroll wheel      Adjust selected speed knot value (+/-0.1)
//  - / =             Decrease / increase selected speed knot value (+/-0.1)
//                      Colour: cyan=slow(<1), white=neutral(1), orange=fast(>1)
//
//  P                 Print current state to stdout
//  S                 Save to level1_edited.db
//
//  Ctrl+C            Copy selected enemy's path points and path speeds
//  Ctrl+V            Paste copied path points and path speeds onto selected enemy
//  Ctrl+Shift+C      Copy entire selected enemy (type, path, cannons, health, etc.)
//  Ctrl+Shift+V      Paste entire enemy onto selected (creates new enemy if none selected)
//
//  Ctrl+Z            Undo last editor action
//  Ctrl+Y            Redo last undone action
// =============================================================================


// Clipboard for copy/paste of entire enemy (Ctrl+Shift+C / Ctrl+Shift+V)
struct EnemyClipboard {
	int                    template_idx = 0;
	std::vector<glm::vec2> path_points;
	std::vector<float>     path_speeds;
	std::vector<cannon>    cannons;
	std::vector<int>       power_ups;
	float                  path_animation_length = 0;
	float                  health = 0;
	float                  max_health = 0;
	int                    path_pixel_delay = 0;
};
EnemyClipboard g_enemy_clipboard;
bool           g_enemy_clipboard_has_data = false;

// ---- Undo / Redo system ------------------------------------------------
//
// Snapshots capture the editable state of every enemy ship.  A snapshot is
// pushed onto g_undoHistory *before* each mutation.  Redo history is cleared
// whenever a new action is performed (standard truncation behaviour).
// Ctrl+Z = undo, Ctrl+Y = redo.
// ------------------------------------------------------------------------

struct CannonSnapshot {
	double min_bullet_interval;
	int    cannon_type;
	double x, y;
};

struct EnemySnapshot {
	int   template_idx;
	float x, y;
	std::vector<glm::vec2>    path_points;
	std::vector<float>        path_speeds;
	std::vector<CannonSnapshot> cannons;
	std::vector<int> power_ups;
	float path_animation_length;
	float health, max_health;
	int   path_pixel_delay;
	float path_scroll_rate;
	float spline_phase_drift;
};

struct EditorUndoState {
	std::vector<EnemySnapshot> enemies;
};

std::vector<EditorUndoState> g_undoHistory;
std::vector<EditorUndoState> g_redoHistory;
const int MAX_UNDO_STEPS = 200;

static EditorUndoState editorCaptureState()
{
	EditorUndoState state;
	state.enemies.reserve(enemy_ships.size());
	for (size_t i = 0; i < enemy_ships.size(); ++i)
	{
		const enemy_ship& e = *enemy_ships[i];
		EnemySnapshot snap;
		snap.template_idx = e.template_idx;
		snap.x = e.x;
		snap.y = e.y;
		snap.path_points = e.path_points;
		snap.path_speeds = e.path_speeds;
		snap.path_animation_length = e.path_animation_length;
		snap.health = e.health;
		snap.max_health = e.max_health;
		snap.path_pixel_delay = e.path_pixel_delay;
		snap.path_scroll_rate = e.path_scroll_rate;
		snap.spline_phase_drift = e.spline_phase_drift;
		snap.cannons.reserve(e.cannons.size());
		for (const auto& c : e.cannons)
		{
			CannonSnapshot cs;
			cs.min_bullet_interval = c.min_bullet_interval;
			cs.cannon_type = c.cannon_type;
			cs.x = c.x;
			cs.y = c.y;
			snap.cannons.push_back(cs);
		}
		snap.power_ups = e.power_ups;
		state.enemies.push_back(std::move(snap));
	}
	return state;
}





static void editorRestoreState(const EditorUndoState& state)
{
	// Shrink: remove excess enemies
	while (enemy_ships.size() > state.enemies.size())
		enemy_ships.pop_back();

	// Grow: create new enemies from their templates
	while (enemy_ships.size() < state.enemies.size())
	{
		int tIdx = state.enemies[enemy_ships.size()].template_idx;
		if (tIdx < 0 || tIdx >= (int)enemy_templates.size()) tIdx = 0;
		enemy_ships.push_back(std::make_unique<enemy_ship>(enemy_templates[tIdx]));
		enemy_ship* ne = enemy_ships.back().get();
		ne->to_be_culled = false;
		ne->manually_update_data(enemy_templates[tIdx].sprite_frames);
		ne->rebuildChunks();
	}

	// Restore every enemy's editable fields
	for (size_t i = 0; i < state.enemies.size(); ++i)
	{
		const EnemySnapshot& snap = state.enemies[i];
		enemy_ship* e = enemy_ships[i].get();

		// Re-apply template if it changed (updates texture, width, height)
		if (e->template_idx != snap.template_idx)
		{
			int tIdx = snap.template_idx;
			if (tIdx >= 0 && tIdx < (int)enemy_templates.size())
			{
				const enemy_ship& tmpl = enemy_templates[tIdx];
				e->template_idx = tIdx;
				e->width = tmpl.width;
				e->height = tmpl.height;
				e->manually_update_data(tmpl.sprite_frames);
				e->rebuildChunks();
			}
		}

		e->x = snap.x;
		e->y = snap.y;
		e->path_points = snap.path_points;
		e->path_speeds = snap.path_speeds;
		e->path_animation_length = snap.path_animation_length;
		e->health = snap.health;
		e->max_health = snap.max_health;
		e->path_pixel_delay = snap.path_pixel_delay;
		e->path_scroll_rate = snap.path_scroll_rate;
		e->spline_phase_drift = snap.spline_phase_drift;

		// Restore cannons
		e->cannons.resize(snap.cannons.size());
		for (size_t j = 0; j < snap.cannons.size(); ++j)
		{
			e->cannons[j].min_bullet_interval = snap.cannons[j].min_bullet_interval;
			e->cannons[j].cannon_type = snap.cannons[j].cannon_type;
			e->cannons[j].x = snap.cannons[j].x;
			e->cannons[j].y = snap.cannons[j].y;
		}

		// Restore power_ups
		e->power_ups = snap.power_ups;
	}

	// Clamp selection indices so nothing is out-of-bounds
	if (!enemy_ships.empty())
		g_selectedEnemy = std::max(0, std::min(g_selectedEnemy, (int)enemy_ships.size() - 1));
	else
		g_selectedEnemy = 0;
	g_selectedPoint = -1;
	g_selectedSpeedKnot = -1;
	g_selectedCannon = -1;
	g_selectedPowerUp = -1;
	g_draggingPoint = false;
}

static void editorPushUndo()
{
	g_undoHistory.push_back(editorCaptureState());
	if ((int)g_undoHistory.size() > MAX_UNDO_STEPS)
		g_undoHistory.erase(g_undoHistory.begin());
	g_redoHistory.clear();
}

static void editorUndo()
{
	if (g_undoHistory.empty()) return;
	g_redoHistory.push_back(editorCaptureState());
	EditorUndoState prev = std::move(g_undoHistory.back());
	g_undoHistory.pop_back();
	editorRestoreState(prev);
	std::cout << "[Editor] Undo  (" << g_undoHistory.size() << " steps remaining)\n";
}

static void editorRedo()
{
	if (g_redoHistory.empty()) return;
	g_undoHistory.push_back(editorCaptureState());
	EditorUndoState next = std::move(g_redoHistory.back());
	g_redoHistory.pop_back();
	editorRestoreState(next);
	std::cout << "[Editor] Redo  (" << g_redoHistory.size() << " steps remaining)\n";
}

static void editorResetUndoHistory()
{
	g_undoHistory.clear();
	g_redoHistory.clear();
}

static bool editorHasEnemy()
{
	return g_editorMode && !enemy_ships.empty();
}

static enemy_ship* editorSelected()
{
	if (!editorHasEnemy()) return nullptr;
	g_selectedEnemy = std::max(0, std::min(g_selectedEnemy, (int)enemy_ships.size() - 1));
	return enemy_ships[g_selectedEnemy].get();
}

// ---- Draw helpers -----------------------------------------------------------

static void editorDrawCircle(float cx, float cy, float r,
	float red, float green, float blue, float alpha)
{
	const int SEG = 24;
	std::vector<Line> segs;
	const float tau = 6.2831853f;
	glm::vec2 prev(cx + r, cy);
	for (int i = 1; i <= SEG; ++i)
	{
		float angle = tau * i / SEG;
		glm::vec2 cur(cx + r * std::cos(angle), cy + r * std::sin(angle));
		segs.push_back(Line(prev, cur, glm::vec4(red, green, blue, alpha)));
		prev = cur;
	}
	drawLinesWithWidth(segs, 2.0f);
}

static void editorDrawCross(float cx, float cy, float half,
	float red, float green, float blue)
{
	std::vector<Line> v;
	glm::vec4 col(red, green, blue, 1.f);
	v.push_back(Line(glm::vec2(cx - half, cy - half), glm::vec2(cx + half, cy + half), col));
	v.push_back(Line(glm::vec2(cx - half, cy + half), glm::vec2(cx + half, cy - half), col));
	drawLinesWithWidth(v, 2.0f);
}

static void editorDrawSpline(const enemy_ship& e, bool isSelected)
{
	if (e.path_points.size() < 2) return;

	glm::vec4 splineCol = isSelected
		? glm::vec4(1.f, 1.f, 0.f, 1.f)
		: glm::vec4(0.5f, 0.5f, 0.8f, 0.6f);

	std::vector<Line> segs;
	glm::vec2 prev = get_spline_point(e.path_points, 0.f);
	const int STEPS = 120;
	for (int i = 1; i <= STEPS; ++i)
	{
		float t = i / float(STEPS);
		glm::vec2 cur = get_spline_point(e.path_points, t);
		segs.push_back(Line(prev, cur, splineCol));
		prev = cur;
	}
	drawLinesWithWidth(segs, isSelected ? 3.f : 1.5f);

	int lastIdx = (int)e.path_points.size() - 1;
	for (size_t i = 0; i < e.path_points.size(); ++i)
	{
		bool isEndpoint = ((int)i == 0 || (int)i == lastIdx);
		bool sel = isSelected && (int)i == g_selectedPoint;
		float r = sel ? 12.f : 8.f;

		// Endpoints live off-screen; clamp X to the screen edge so they're clickable
		float drawX = isEndpoint
			? std::max(r, std::min(e.path_points[i].x, (float)windowWidth - r))
			: e.path_points[i].x;
		float drawY = e.path_points[i].y;

		// Cyan for endpoints (X is locked), yellow/orange for interior points
		float cr = isEndpoint ? 0.f : (sel ? 1.f : 0.8f);
		float cg = isEndpoint ? (sel ? 1.f : 0.8f) : (sel ? 0.6f : 0.8f);
		float cb = isEndpoint ? (sel ? 1.f : 0.8f) : 0.f;

		editorDrawCircle(drawX, drawY, r, cr, cg, cb, 1.f);

		// Draw a small horizontal arrow on endpoints to indicate X is locked
		if (isEndpoint && isSelected)
		{
			std::vector<Line> arr;
			float dir = ((int)i == 0) ? -1.f : 1.f;  // arrow points off-screen
			arr.push_back(Line(glm::vec2(drawX, drawY),
				glm::vec2(drawX + dir * 20.f, drawY),
				glm::vec4(cr, cg, cb, 0.7f)));
			drawLinesWithWidth(arr, 2.f);
		}
	}

	// Draw width x height boxes centred on the first and last knots
	if (e.path_points.size() >= 2)
	{
		glm::vec4 boxCol = isSelected
			? glm::vec4(0.f, 1.f, 1.f, 0.8f)
			: glm::vec4(0.f, 0.6f, 0.6f, 0.5f);

		int endpoints[2] = { 0, (int)e.path_points.size() - 1 };
		for (int ei = 0; ei < 2; ++ei)
		{
			float cx = e.path_points[endpoints[ei]].x;
			float cy = e.path_points[endpoints[ei]].y;
			float hw = e.width * 0.5f;
			float hh = e.height * 0.5f;
			float bx0 = cx - hw, by0 = cy - hh;
			float bx1 = cx + hw, by1 = cy + hh;

			std::vector<Line> box;
			box.push_back(Line(glm::vec2(bx0, by0), glm::vec2(bx1, by0), boxCol));
			box.push_back(Line(glm::vec2(bx1, by0), glm::vec2(bx1, by1), boxCol));
			box.push_back(Line(glm::vec2(bx1, by1), glm::vec2(bx0, by1), boxCol));
			box.push_back(Line(glm::vec2(bx0, by1), glm::vec2(bx0, by0), boxCol));
			drawLinesWithWidth(box, 2.f);
		}
	}
}

static void editorDrawCannons(const enemy_ship& e)
{
	for (size_t ci = 0; ci < e.cannons.size(); ++ci)
	{
		float wx = e.x + (float)e.cannons[ci].x;
		float wy = e.y + (float)e.cannons[ci].y;

		float cr = 1.f, cg = 0.0f, cb = 0.0f;   // LEFT  = red
		if (e.cannons[ci].cannon_type == CANNON_TYPE_UP_DOWN) { cr = 0.0f; cg = 1.0f; cb = 0.0f; }
		if (e.cannons[ci].cannon_type == CANNON_TYPE_TRACKING) { cr = 0.0f; cg = 0.0f; cb = 1.0f; }
		if (e.cannons[ci].cannon_type == CANNON_TYPE_CIRCULAR) { cr = 1.0f; cg = 1.0f; cb = 0.0f; }

		bool sel = ((int)ci == g_selectedCannon);
		float radius = 20.0f;

		editorDrawCircle(wx, wy, radius, cr, cg, cb, 1.f);
		editorDrawCross(wx, wy, radius, cr, cg, cb);

		// Draw a bright yellow selection ring around the active cannon
		if (sel)
			editorDrawCircle(wx, wy, radius + 7.f, 1.f, 1.f, 0.f, 0.9f);
	}
}

// Returns the index of the speed knot whose spline-position is closest to (mx,my),
// or -1 if none is within the given pixel threshold.
static int editorFindNearestSpeedKnot(const enemy_ship& e, float mx, float my,
	float threshold = 25.f)
{
	int n = (int)e.path_speeds.size();
	if (n == 0 || e.path_points.size() < 2) return -1;

	int best = -1;
	float bestDist2 = threshold * threshold;
	for (int i = 0; i < n; ++i)
	{
		float t = (n == 1) ? 0.5f : (float)i / (float)(n - 1);
		glm::vec2 pos = get_spline_point(e.path_points, t);
		float dx = pos.x - mx, dy = pos.y - my;
		float d2 = dx * dx + dy * dy;
		if (d2 < bestDist2) { bestDist2 = d2; best = i; }
	}
	return best;
}

// Draw diamond markers for each speed knot along the selected enemy's spline.
// Color encodes value: cyan = slow (<1), white = neutral (=1), orange = fast (>1).
// The selected knot is drawn larger and in bright yellow.
static void editorDrawSpeedKnots(const enemy_ship& e, int selectedKnot)
{
	int n = (int)e.path_speeds.size();
	if (n == 0 || e.path_points.size() < 2) return;

	for (int i = 0; i < n; ++i)
	{
		float t = (n == 1) ? 0.5f : (float)i / (float)(n - 1);
		glm::vec2 pos = get_spline_point(e.path_points, t);
		float speed = e.path_speeds[i];
		bool sel = (i == selectedKnot);

		// Colour ramp: blue(0) -> white(1) -> orange(2+)
		float cr, cg, cb;
		if (speed <= 1.0f)
		{
			float s = std::max(0.f, speed);   // 0..1
			cr = s; cg = s; cb = 1.f;         // dark-blue -> white
		}
		else
		{
			float s = std::min(speed - 1.0f, 1.0f);  // 0..1 for speed 1..2
			cr = 1.f; cg = 1.f - s * 0.7f; cb = 0.f; // white -> orange
		}

		float half = sel ? 14.f : 10.f;
		glm::vec4 col = sel ? glm::vec4(1.f, 1.f, 0.f, 1.f)
			: glm::vec4(cr, cg, cb, 1.f);

		// Diamond outline
		glm::vec2 top(pos.x, pos.y - half);
		glm::vec2 right(pos.x + half, pos.y);
		glm::vec2 bot(pos.x, pos.y + half);
		glm::vec2 left(pos.x - half, pos.y);

		std::vector<Line> diamond;
		diamond.push_back(Line(top, right, col));
		diamond.push_back(Line(right, bot, col));
		diamond.push_back(Line(bot, left, col));
		diamond.push_back(Line(left, top, col));
		drawLinesWithWidth(diamond, sel ? 3.f : 2.f);

		// Extra inner diamond for selected knot
		if (sel)
		{
			float ih = half * 0.45f;
			glm::vec4 innerCol(1.f, 1.f, 0.f, 0.45f);
			std::vector<Line> inner;
			inner.push_back(Line(glm::vec2(pos.x, pos.y - ih), glm::vec2(pos.x + ih, pos.y), innerCol));
			inner.push_back(Line(glm::vec2(pos.x + ih, pos.y), glm::vec2(pos.x, pos.y + ih), innerCol));
			inner.push_back(Line(glm::vec2(pos.x, pos.y + ih), glm::vec2(pos.x - ih, pos.y), innerCol));
			inner.push_back(Line(glm::vec2(pos.x - ih, pos.y), glm::vec2(pos.x, pos.y - ih), innerCol));
			drawLinesWithWidth(inner, 1.5f);
		}
	}
}

static void editorDrawSelectionBox(const enemy_ship& e)
{
	float x0 = e.x, y0 = e.y;
	float x1 = x0 + e.width, y1 = y0 + e.height;
	glm::vec4 col(1, 1, 0, 1);
	std::vector<Line> v;
	v.push_back(Line(glm::vec2(x0, y0), glm::vec2(x1, y0), col));
	v.push_back(Line(glm::vec2(x1, y0), glm::vec2(x1, y1), col));
	v.push_back(Line(glm::vec2(x1, y1), glm::vec2(x0, y1), col));
	v.push_back(Line(glm::vec2(x0, y1), glm::vec2(x0, y0), col));
	drawLinesWithWidth(v, 2.f);
}

// ---- Time ruler -------------------------------------------------------------
// Draws vertical lines at one-second intervals of gameplay scroll. During
// gameplay the foreground moves at foreground_vel px/s, so lines are spaced
// |foreground_vel| pixels apart. They scroll with the editor view via
// g_editorScrollAccum so the line labeled "Ns" always marks the world position
// the foreground will occupy N seconds into the level.
static void editorDrawTimeRuler()
{
	if (draw_time_lines == false)
		return;



	// No meaningful ruler if the foreground isn't scrolling.
	if (std::fabs(foreground_vel) < 1e-3f) return;

	// Pixels-per-second of gameplay scroll (always positive for spacing math).
	const float pxPerSec = std::fabs(foreground_vel);

	// Screen-x of the T-second marker:
	//   screen_x(T) = foreground_vel * T + g_editorScrollAccum
	// Solve for the range of T that falls within [0, windowWidth].
	float t_at_left = (0.0f - g_editorScrollAccum) / foreground_vel;
	float t_at_right = ((float)windowWidth - g_editorScrollAccum) / foreground_vel;
	float t_min = std::min(t_at_left, t_at_right);
	float t_max = std::max(t_at_left, t_at_right);

	int iMin = (int)std::floor(t_min);
	int iMax = (int)std::ceil(t_max);

	// Sanity clamp so a weird scroll value can't spawn thousands of lines.
	const int MAX_LINES = 4096;
	if (iMax - iMin > MAX_LINES) return;

	// Styling
	const glm::vec4 colTick(0.55f, 0.75f, 0.95f, 0.2f);  // every second
	const glm::vec4 colZero(1.00f, 0.55f, 0.20f, 0.2f);  // T = 0 (level start)
	const glm::vec4 colLabel(0.85f, 0.95f, 1.00f, 0.95f);
	const glm::vec4 colLabelZero(1.00f, 0.75f, 0.35f, 1.00f);

	// Batch ticks and the T=0 line separately so each can have its own width.
	std::vector<Line> tickLines;
	std::vector<Line> zeroLines;

	const float topY = 0.0f;
	const float botY = (float)windowHeight;
	const float labelY = 28.0f;         // top label
	const float labelY2 = botY - 60.0f;  // bottom label (so it's visible even if HUD covers top)

	for (int T = iMin; T <= iMax; ++T)
	{
		float x = foreground_vel * (float)T + g_editorScrollAccum;
		if (x < -1.0f || x >(float)windowWidth + 1.0f) continue;

		if (T == 0)
			zeroLines.push_back(Line(glm::vec2(x, topY), glm::vec2(x, botY), colZero));
		else
			tickLines.push_back(Line(glm::vec2(x, topY), glm::vec2(x, botY), colTick));
	}

	if (!tickLines.empty()) drawLinesWithWidth(tickLines, 1.5f);
	if (!zeroLines.empty()) drawLinesWithWidth(zeroLines, 3.0f);

	// Labels: one per second. If the spacing is too tight for per-second
	// labels without overlap (~40px wide for "00s" at scale 0.35), fall back
	// to every-5th-second labels plus T=0 so the numbers stay readable.
	const bool labelEverySecond = (pxPerSec >= 40.0f);

	if (textRenderer)
	{
		for (int T = iMin; T <= iMax; ++T)
		{
			float x = foreground_vel * (float)T + g_editorScrollAccum;
			if (x < 0.0f || x >(float)windowWidth) continue;

			bool isZero = (T == 0);
			if (!isZero && !labelEverySecond && (T % 5 != 0)) continue;

			char buf[32];
			snprintf(buf, sizeof(buf), "%d", T);

			glm::vec4 col = isZero ? colLabelZero : colLabel;
			float scale = 0.25;//isZero ? 0.45f : 0.38f;

			// Nudge label a couple of pixels right of the line so it doesn't
			// sit directly on top.
			//textRenderer->renderText(buf, x + 3.0f, labelY, scale, col);
			textRenderer->renderText(buf, x + 3.0f, labelY2, scale, col);
		}
	}
}

// ---- Main overlay render (called inside display()) --------------------------

void renderEditorOverlay()
{
	if (!g_editorMode) return;

	// Draw the time ruler first so spline/sprite overlays paint on top of it.
	editorDrawTimeRuler();

	for (size_t i = 0; i < enemy_ships.size(); ++i)
	{
		bool sel = ((int)i == g_selectedEnemy);
		editorDrawSpline(*enemy_ships[i], sel);
		if (sel)
		{
			editorDrawSelectionBox(*enemy_ships[i]);
			editorDrawCannons(*enemy_ships[i]);
			editorDrawSpeedKnots(*enemy_ships[i], g_selectedSpeedKnot);
		}
	}

	if (textRenderer)
	{
		enemy_ship* e = editorSelected();

		char buf[512];
		snprintf(buf, sizeof(buf), "Enemies:%d  Sel:%d",
			(int)enemy_ships.size(), g_selectedEnemy);
		textRenderer->renderText(buf, 10, 10, 0.5f, glm::vec4(1, 1, 0, 1));

		if (e)
		{
			ostringstream oss = editorPrintState(g_selectedEnemy);

			std::vector<std::string> result;
			std::string line;
			std::istringstream stream(oss.str());

			while (std::getline(stream, line)) {
				result.push_back(line);
			}

			for (size_t i = 0; i < result.size(); i++)
			{
				// Lines prefixed with ">>" are the currently-selected
				// cannon or power-up — render them in a brighter highlight
				bool highlighted = (result[i].find(">>") != std::string::npos);
				glm::vec4 col = highlighted
					? glm::vec4(1.f, 1.f, 1.f, 1.f)     // white = selected
					: glm::vec4(1.f, 1.f, 0.f, 1.f);    // yellow = normal
				textRenderer->renderText(result[i].c_str(), 10, (i + 1) * 50.0f, 0.5f, col);
			}

			// ---- Speed-knot value labels (drawn next to each diamond marker) ----
			int sn = (int)e->path_speeds.size();
			if (sn > 0 && e->path_points.size() >= 2)
			{
				for (int si = 0; si < sn; ++si)
				{
					float t = (sn == 1) ? 0.5f : (float)si / (float)(sn - 1);
					glm::vec2 pos = get_spline_point(e->path_points, t);
					bool sel = (si == g_selectedSpeedKnot);

					char lbl[64];
					snprintf(lbl, sizeof(lbl), "%.2f", e->path_speeds[si]);

					glm::vec4 lblCol = sel
						? glm::vec4(1.f, 1.f, 0.f, 1.f)
						: glm::vec4(0.6f, 0.9f, 1.f, 0.85f);

					textRenderer->renderText(lbl, pos.x + 16.f, pos.y - 10.f,
						sel ? 0.45f : 0.38f, lblCol);
				}
			}

			// ---- Selected speed-knot status line --------------------------------
			if (g_selectedSpeedKnot >= 0 && g_selectedSpeedKnot < sn)
			{
				//snprintf(buf, sizeof(buf),
				//	"Speed knot [%d/%d] = %.2f   Scroll or -/= to adjust",
				//	g_selectedSpeedKnot, sn - 1,
				//	e->path_speeds[g_selectedSpeedKnot]);
				//textRenderer->renderText(buf, 10, (float)windowHeight - 60,
				//	0.5f, glm::vec4(1.f, 1.f, 0.f, 1.f));
			}
			else
			{
				// Generic speed-knot hint when none is selected
				if (sn > 0)
				{
					//snprintf(buf, sizeof(buf),
					//	"Speed knots: %d  |  LMB diamond to select  |  Shift+LMB add  |  Shift+RMB remove",
					//	sn);
					//textRenderer->renderText(buf, 10, (float)windowHeight - 60,
					//	0.45f, glm::vec4(0.6f, 0.9f, 1.f, 0.8f));
				}
			}

			// ---- Selected cannon status line ------------------------------------
			int cn = (int)e->cannons.size();
			if (cn > 0)
			{
				if (g_selectedCannon >= 0 && g_selectedCannon < cn)
				{
					//const cannon& sc = e->cannons[g_selectedCannon];
					//const char* typeNames[] = { "LEFT", "UP_DOWN", "TRACKING", "CIRCULAR" };
					//snprintf(buf, sizeof(buf),
					//	"Cannon [%d/%d]  type=%s  interval=%.2fs   T=cycle type  ,/.=interval  V=remove",
					//	g_selectedCannon, cn - 1,
					//	typeNames[sc.cannon_type],
					//	sc.min_bullet_interval);
					//textRenderer->renderText(buf, 10, (float)windowHeight - 110,
					//	0.45f, glm::vec4(1.f, 0.6f, 0.3f, 1.f));
				}
				else
				{
					//snprintf(buf, sizeof(buf),
					//	"Cannons: %d  |  { / } to select  |  C=add  V=remove  T=type  ,/.=interval",
					//	cn);
					//textRenderer->renderText(buf, 10, (float)windowHeight - 110,
					//	0.45f, glm::vec4(1.f, 0.6f, 0.3f, 0.8f));
				}
			}

			// ---- Selected power-up status line ----------------------------------
			int pn = (int)e->power_ups.size();
			if (pn > 0)
			{
				const char* puNames[] = { "SINUSOIDAL", "X3", "X5" };
				if (g_selectedPowerUp >= 0 && g_selectedPowerUp < pn)
				{
					//int t = e->power_ups[g_selectedPowerUp];
					//snprintf(buf, sizeof(buf),
					//	"Power-up [%d/%d]  type=%s   Y=cycle type  F=remove  B=add",
					//	g_selectedPowerUp, pn - 1,
					//	(t >= 0 && t < NUM_POWER_UP_TYPES) ? puNames[t] : "?");
					//textRenderer->renderText(buf, 10, (float)windowHeight - 160,
					//	0.45f, glm::vec4(0.6f, 1.f, 0.4f, 1.f));
				}
				else
				{
					//snprintf(buf, sizeof(buf),
					//	"Power-ups: %d  |  : / ' to select  |  B=add  F=remove  Y=type",
					//	pn);
					//textRenderer->renderText(buf, 10, (float)windowHeight - 160,
					//	0.45f, glm::vec4(0.6f, 1.f, 0.4f, 0.8f));
				}
			}
			else
			{
				// No power-ups yet — show a hint so the user knows the feature exists
				//snprintf(buf, sizeof(buf),
				//	"Power-ups: 0  |  B=add a power-up to this enemy");
				//textRenderer->renderText(buf, 10, (float)windowHeight - 160,
				//	0.4f, glm::vec4(0.6f, 1.f, 0.4f, 0.6f));
			}
		}
	}
}

// ---- Path-point search ------------------------------------------------------

static int editorFindNearestPoint(const enemy_ship& e, float mx, float my,
	float threshold = 20.f)
{
	int best = -1;
	float bestDist = threshold * threshold;
	int lastIdx = (int)e.path_points.size() - 1;
	for (size_t i = 0; i < e.path_points.size(); ++i)
	{
		bool isEndpoint = ((int)i == 0 || (int)i == lastIdx);
		// Endpoints are drawn clamped to the screen edge — use the same position for picking
		float testX = isEndpoint
			? std::max(0.f, std::min(e.path_points[i].x, (float)windowWidth))
			: e.path_points[i].x;
		float dx = testX - mx;
		float dy = e.path_points[i].y - my;
		float d2 = dx * dx + dy * dy;
		if (d2 < bestDist) { bestDist = d2; best = (int)i; }
	}
	return best;
}

// ---- Print state to stdout --------------------------------------------------


// ---- Save to SQLite ---------------------------------------------------------

// Creates the canonical level schema on `db` if it does not already exist,
// and seeds the static lookup tables (cannon_type, power_up).  Safe to call
// repeatedly: all CREATE statements use IF NOT EXISTS and all seed INSERTs
// use OR IGNORE against explicit primary keys, so existing data is preserved.
static void ensureDatabaseSchema(sqlite3* db)
{
	auto exec = [&](const char* sql) {
		char* err = nullptr;
		if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK)
		{
			std::cerr << "[DB] Schema SQL error (" << sql << "): " << err << "\n";
			sqlite3_free(err);
		}
		};

	exec(R"(CREATE TABLE IF NOT EXISTS one_d_location (
		one_d_location_id INTEGER PRIMARY KEY NOT NULL,
		x REAL NOT NULL,
		UNIQUE (x)
	);)");

	exec(R"(CREATE TABLE IF NOT EXISTS two_d_location (
		two_d_location_id INTEGER PRIMARY KEY NOT NULL,
		x REAL NOT NULL,
		y REAL NOT NULL,
		UNIQUE (x, y)
	);)");

	exec(R"(CREATE TABLE IF NOT EXISTS path (
		path_id INTEGER PRIMARY KEY NOT NULL,
		path_animation_length REAL NOT NULL,
		path_nickname TEXT
	);)");

	exec(R"(CREATE TABLE IF NOT EXISTS path_speed (
		path_speed_id INTEGER PRIMARY KEY NOT NULL,
		path_id INTEGER NOT NULL,
		one_d_location_id INTEGER NOT NULL,
		FOREIGN KEY (path_id) REFERENCES path(path_id),
		FOREIGN KEY (one_d_location_id) REFERENCES one_d_location(one_d_location_id),
		UNIQUE (path_speed_id, path_id, one_d_location_id)
	);)");

	exec(R"(CREATE TABLE IF NOT EXISTS path_location (
		path_location_id INTEGER PRIMARY KEY NOT NULL,
		path_id INTEGER NOT NULL,
		two_d_location_id INTEGER NOT NULL,
		FOREIGN KEY (path_id) REFERENCES path(path_id),
		FOREIGN KEY (two_d_location_id) REFERENCES two_d_location(two_d_location_id),
		UNIQUE (path_location_id, path_id, two_d_location_id)
	);)");

	exec(R"(CREATE TABLE IF NOT EXISTS enemy (
		enemy_id INTEGER PRIMARY KEY NOT NULL,
		file_template_id INTEGER NOT NULL,
		path_id INTEGER NOT NULL,
		path_pixel_delay INTEGER NOT NULL,
		max_health REAL NOT NULL,
		enemy_nickname TEXT,
		FOREIGN KEY (path_id) REFERENCES path(path_id)
	);)");

	exec(R"(CREATE TABLE IF NOT EXISTS cannon_type (
		cannon_type_id INTEGER PRIMARY KEY NOT NULL,
		cannon_type_nickname TEXT
	);)");

	// Seed cannon_type with the canonical 4 rows.  Explicit IDs + OR IGNORE
	// make this idempotent; get_cannons() does cannon_type -= 1 on read, so
	// IDs must remain 1..4.
	exec("INSERT OR IGNORE INTO cannon_type (cannon_type_id, cannon_type_nickname) VALUES (1, 'left shot');");
	exec("INSERT OR IGNORE INTO cannon_type (cannon_type_id, cannon_type_nickname) VALUES (2, 'up-down shot');");
	exec("INSERT OR IGNORE INTO cannon_type (cannon_type_id, cannon_type_nickname) VALUES (3, 'tracking shot');");
	exec("INSERT OR IGNORE INTO cannon_type (cannon_type_id, cannon_type_nickname) VALUES (4, 'circular shot');");

	exec(R"(CREATE TABLE IF NOT EXISTS enemy_cannon (
		enemy_cannon_id INTEGER PRIMARY KEY NOT NULL,
		cannon_type_id INTEGER NOT NULL,	
		enemy_id INTEGER NOT NULL,
		two_d_location_id INTEGER NOT NULL,
		min_bullet_interval REAL NOT NULL,
		enemy_cannon_nickname TEXT,
		FOREIGN KEY (cannon_type_id) REFERENCES cannon_type(cannon_type_id),
		FOREIGN KEY (enemy_id) REFERENCES enemy(enemy_id),
		FOREIGN KEY (two_d_location_id) REFERENCES two_d_location(two_d_location_id)
	);)");

	exec(R"(CREATE TABLE IF NOT EXISTS power_up (
		power_up_id INTEGER PRIMARY KEY NOT NULL,
		power_up_nickname TEXT
	);)");

	exec("INSERT OR IGNORE INTO power_up (power_up_id, power_up_nickname) VALUES (1, 'sinusoidal fire');");
	exec("INSERT OR IGNORE INTO power_up (power_up_id, power_up_nickname) VALUES (2, 'x3 fire');");
	exec("INSERT OR IGNORE INTO power_up (power_up_id, power_up_nickname) VALUES (3, 'x5 fire');");

	exec(R"(CREATE TABLE IF NOT EXISTS enemy_power_up (
		enemy_power_up_id INTEGER PRIMARY KEY NOT NULL,
		power_up_id INTEGER NOT NULL,
		enemy_id INTEGER NOT NULL,
		FOREIGN KEY(power_up_id) REFERENCES power_up(power_up_id),
		FOREIGN KEY(enemy_id) REFERENCES enemy(enemy_id)
	);)");
}

static void editorSaveToDatabase(const std::string& db_name)
{
	sqlite3* db = nullptr;
	if (sqlite3_open(db_name.c_str(), &db) != SQLITE_OK)
	{
		std::cerr << "[Editor] Cannot open DB: " << sqlite3_errmsg(db) << "\n";
		return;
	}

	auto exec = [&](const char* sql) {
		char* err = nullptr;
		if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK)
		{
			std::cerr << "[Editor] SQL error (" << sql << "): " << err << "\n";

			std::cerr << sql << '\n';

			sqlite3_free(err);
		}
		};

	exec("BEGIN TRANSACTION;");

	// Drop all tables so we always start from the canonical schema,
	// regardless of what was on disk before.  Order respects FK deps.

	exec("DROP TABLE IF EXISTS enemy_power_up;");
	exec("DROP TABLE IF EXISTS power_up;");
	exec("DROP TABLE IF EXISTS enemy_cannon;");
	exec("DROP TABLE IF EXISTS cannon_type;");
	exec("DROP TABLE IF EXISTS enemy;");
	exec("DROP TABLE IF EXISTS path_location;");
	exec("DROP TABLE IF EXISTS path_speed;");
	exec("DROP TABLE IF EXISTS path;");
	exec("DROP TABLE IF EXISTS two_d_location;");
	exec("DROP TABLE IF EXISTS one_d_location;");



	// Recreate the canonical schema and reseed lookup tables.  Tables were
	// just dropped above, so IF NOT EXISTS is a no-op on first hit; the same
	// helper is also called on load to bootstrap an empty database file.
	ensureDatabaseSchema(db);





	int path_id = 0;
	int path_location_id = 0;
	int path_speed_id = 0;
	int loc2d_id = 0;
	int loc1d_id = 0;
	int enemy_id = 0;
	int cannon_id = 0;
	int enemy_power_up_id = 0;

	// Returns the two_d_location_id for (x,y), inserting a new row only when
	// that coordinate pair is not already present (satisfies UNIQUE(x,y)).
	auto use2D = [&](float x, float y) -> int {
		char q[256];
		snprintf(q, sizeof(q),
			"SELECT two_d_location_id FROM two_d_location WHERE x=%.6f AND y=%.6f;",
			x, y);
		int found = -1;
		sqlite3_exec(db, q, [](void* d, int, char** v, char**) -> int {
			*static_cast<int*>(d) = std::atoi(v[0]); return 0;
			}, &found, nullptr);
		if (found != -1) return found;
		loc2d_id++;
		char s[256];
		snprintf(s, sizeof(s),
			"INSERT INTO two_d_location(two_d_location_id,x,y) VALUES(%d,%.6f,%.6f);",
			loc2d_id, x, y);
		exec(s);
		return loc2d_id;
		};

	// Returns the one_d_location_id for x, inserting only when absent (UNIQUE(x)).
	auto use1D = [&](float x) -> int {
		char q[256];
		snprintf(q, sizeof(q),
			"SELECT one_d_location_id FROM one_d_location WHERE x=%.6f;", x);
		int found = -1;
		sqlite3_exec(db, q, [](void* d, int, char** v, char**) -> int {
			*static_cast<int*>(d) = std::atoi(v[0]); return 0;
			}, &found, nullptr);
		if (found != -1) return found;
		loc1d_id++;
		char s[256];
		snprintf(s, sizeof(s),
			"INSERT INTO one_d_location(one_d_location_id,x) VALUES(%d,%.6f);",
			loc1d_id, x);
		exec(s);
		return loc1d_id;
		};

	for (size_t i = 0; i < enemy_ships.size(); ++i)
	{
		const enemy_ship& e = *enemy_ships[i];

		// path row
		path_id++;
		{
			char sql[256];
			snprintf(sql, sizeof(sql),
				"INSERT INTO path(path_id, path_animation_length) VALUES(%d, %.4f);",
				path_id, e.path_animation_length);
			exec(sql);
		}

		// path control-point locations
		for (size_t j = 0; j < e.path_points.size(); ++j)
		{
			// path_points[j].x is in world space with path_pixel_delay
			// AND any editor arrow-key scroll applied since load baked in.
			// Strip both to get the canonical, scroll-independent position.
			//
			// path_pixel_delay is immutable across save/load and is reapplied
			// at load time, so we only remove it here, not modify it.
			//
			// g_editorScrollAccum is the sum of every arrow-key scrollDelta
			// applied to path_points in editor mode since load. The foreground
			// tiles are not persisted, so on the next load they will be
			// reconstructed at their original positions; if we did not strip
			// the editor delta here, every enemy would be shifted by exactly
			// that delta relative to a foreground that no longer has it.
			//
			// Gameplay drift is NOT subtracted: while the level is being
			// edited the simulation is paused, so no gameplay-drift offset
			// can accumulate into path_points between load and save.



			float fg_scroll = foreground_chunked.empty()
				? 0.0f
				: (foreground_chunked[0].x - g_loadTimeFgX);



			// During the active spline phase, path_points are frozen in
			// screen space while the foreground continues to drift. The
			// raw fg_scroll term over-corrects by exactly the amount of
			// drift the path missed; spline_phase_drift records that
			// missed amount per enemy and we add it back here so the
			// canonical x round-trips. (For enemies that never entered
			// the spline phase, spline_phase_drift is zero and this is
			// a no-op.)
			float nx = (e.path_points[j].x
				- static_cast<float>(e.path_pixel_delay)
				//- g_editorScrollAccum
				- fg_scroll
				+ e.spline_phase_drift) / SIM_WIDTH;

			float ny = e.path_points[j].y / SIM_HEIGHT;




			int tid = use2D(nx, ny);

			path_location_id++;
			char sql[256];
			snprintf(sql, sizeof(sql),
				"INSERT INTO path_location(path_location_id,path_id,two_d_location_id)"
				" VALUES(%d,%d,%d);",
				path_location_id, path_id, tid);
			exec(sql);
		}

		// speed knots
		for (size_t j = 0; j < e.path_speeds.size(); ++j)
		{
			int oid = use1D(e.path_speeds[j]);

			path_speed_id++;
			char sql[256];
			snprintf(sql, sizeof(sql),
				"INSERT INTO path_speed(path_speed_id,path_id,one_d_location_id)"
				" VALUES(%d,%d,%d);",
				path_speed_id, path_id, oid);
			exec(sql);
		}

		// enemy row
		enemy_id++;
		{
			int tpl_id = 0;
			for (size_t t = 0; t < enemy_templates.size(); ++t)
				if (enemy_templates[t].width == e.width && enemy_templates[t].height == e.height)
				{
					tpl_id = (int)t; break;
				}

			char sql[256];
			snprintf(sql, sizeof(sql),
				"INSERT INTO enemy(enemy_id,file_template_id,path_id,path_pixel_delay,max_health)"
				" VALUES(%d,%d,%d,%d,%.1f);",
				enemy_id, tpl_id, path_id, e.path_pixel_delay, e.max_health);
			exec(sql);
		}

		// cannons
		for (size_t j = 0; j < e.cannons.size(); ++j)
		{
			const cannon& c = e.cannons[j];
			float cx = (e.width > 1) ? (float)(c.x / (e.width - 1)) : 0.0f;
			float cy = (e.height > 1) ? (float)(c.y / (e.height - 1)) : 0.0f;
			int tid = use2D(cx, cy);

			cannon_id++;
			char sql[256];
			snprintf(sql, sizeof(sql),
				"INSERT INTO enemy_cannon(enemy_cannon_id,enemy_id,cannon_type_id,"
				"two_d_location_id,min_bullet_interval)"
				" VALUES(%d,%d,%d,%d,%.4f);",
				cannon_id, enemy_id,
				c.cannon_type + 1,  // back to 1-based
				tid,
				c.min_bullet_interval);
			exec(sql);
		}

		// power-ups
		for (size_t j = 0; j < e.power_ups.size(); ++j)
		{
			enemy_power_up_id++;
			char sql[256];
			snprintf(sql, sizeof(sql),
				"INSERT INTO enemy_power_up(enemy_power_up_id,power_up_id,enemy_id)"
				" VALUES(%d,%d,%d);",
				enemy_power_up_id,
				e.power_ups[j] + 1,  // back to 1-based
				enemy_id);
			exec(sql);
		}
	}

	exec("COMMIT;");
	sqlite3_close(db);
	std::cout << "[Editor] Saved " << enemy_ships.size()
		<< " enemies to " << db_name << "\n";
}

// ---- Keyboard handler (returns true if editor consumed the key) -------------

// Return which template index the given enemy currently uses (match by tex handle)


// Swap the sprite/GL data of an existing enemy to a different template,
// keeping its path, cannons, position, and health intact.
static void editorApplyTemplate(enemy_ship* e, int tIdx)
{
	if (tIdx < 0 || tIdx >= (int)enemy_templates.size()) return;
	const enemy_ship& tmpl = enemy_templates[tIdx];

	// Remember centre position so the ship doesn't jump when size changes
	float cx = e->x + e->width * 0.5f;
	float cy = e->y + e->height * 0.5f;

	// Capture the old half-width before we overwrite it; used to fix endpoints below.
	float old_half_w = e->width * 0.5f;

	// Preserve everything that isn't visual
	auto  saved_path_points = e->path_points;
	auto  saved_path_speeds = e->path_speeds;
	auto  saved_cannons = e->cannons;
	float saved_path_t = e->path_t;
	float saved_path_animation_length = e->path_animation_length;
	float saved_health = e->health;
	float saved_max_health = e->max_health;

	// Apply new visual from template
	e->template_idx = tIdx;
	e->tex = tmpl.tex;
	e->width = tmpl.width;
	e->height = tmpl.height;
	e->manually_update_data(tmpl.sprite_frames);
	e->rebuildChunks();

	// Re-centre the sprite on screen
	e->x = cx - e->width * 0.5f;
	e->y = cy - e->height * 0.5f;

	// Restore gameplay state
	e->path_points = saved_path_points;
	e->path_speeds = saved_path_speeds;
	e->path_t = saved_path_t;
	e->path_animation_length = saved_path_animation_length;
	e->health = saved_health;
	e->max_health = saved_max_health;

	// Fix the endpoint X positions to reflect the new half-width.
	//
	// When loaded / spawned the endpoints are set to:
	//   path_points[0].x    = SIM_WIDTH + half_w  (+ pixel_delay offset)
	//   path_points[last].x = -half_w             (+ pixel_delay offset)
	//
	// path_pixel_delay is anchored to the LEFTMOST (last/exit) knot:
	//   delay = path_points.back().x - (-half_w)
	//
	// If the template size changed we must slide them by the delta so the
	// enemy still enters and exits fully off-screen.
	if (e->path_points.size() >= 2)
	{
		float new_half_w = e->width * 0.5f;
		float delta = new_half_w - old_half_w;

		// First endpoint moves further right when the sprite is wider.
		e->path_points.front().x += delta;

		// Last endpoint moves further left when the sprite is wider.
		e->path_points.back().x -= delta;

		// Keep path_pixel_delay consistent: it is anchored to the
		// leftmost (exit) knot, whose canonical position is -new_half_w.
		// delay = path_points.back().x - (-new_half_w), clamped to >= 0.
		e->path_pixel_delay = std::max(0, (int)(e->path_points.back().x + new_half_w));
	}

	// Always recalculate path_scroll_rate (works for both active and pre-activation enemies).
	if (e->path_animation_length > 0.0f)
	{
		float actual_duration = calculate_actual_path_duration(
			e->path_points, e->path_speeds, e->path_animation_length);
		e->path_scroll_rate = -(e->width * 0.5f) / actual_duration;
	}

	// Restore cannons, clamping local coords to the new sprite bounds
	e->cannons = saved_cannons;
	for (auto& c : e->cannons)
	{
		c.x = std::max(0.0, std::min(c.x, (double)(e->width - 1)));
		c.y = std::max(0.0, std::min(c.y, (double)(e->height - 1)));
	}
}





vector<float> get_path_speeds(int path_id, sqlite3* (&db))
{
	size_t row_count = 0;

	vector<float> path_speeds;

	sqlite3_stmt* stmt;

	ostringstream oss;
	//oss << "SELECT t.x FROM path_speed ps JOIN one_d_location t ON ps.one_d_location_id = t.one_d_location_id WHERE ps.path_id = " << path_id << ";";

	oss << "SELECT t.x FROM path_speed ps "
		"JOIN one_d_location t ON ps.one_d_location_id = t.one_d_location_id "
		"WHERE ps.path_id = " << path_id << " "
		"ORDER BY ps.path_speed_id;";


	int rc = sqlite3_prepare_v2(db, oss.str().c_str(), -1, &stmt, nullptr);

	if (rc != SQLITE_OK)
	{
		std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
		return path_speeds;
	}

	bool done = false;

	while (!done)
	{
		switch (sqlite3_step(stmt))
		{
		case SQLITE_ROW:
		{
			float x = static_cast<float>(sqlite3_column_double(stmt, 0));

			path_speeds.push_back(x);

			row_count++;

			break;
		}
		case SQLITE_DONE:
		{
			done = true;
			break;
		}
		default:
		{
			done = true;
			cout << "Failure" << endl;
			break;
		}
		}
	}

	sqlite3_finalize(stmt);
	return path_speeds;
}


vector<glm::vec2> get_path_points(int path_id, sqlite3* (&db))
{
	size_t row_count = 0;

	vector<glm::vec2> path_points;

	sqlite3_stmt* stmt;

	ostringstream oss;
	//oss << "SELECT t.x, t.y FROM path_location pl JOIN two_d_location t ON pl.two_d_location_id = t.two_d_location_id WHERE pl.path_id = " << path_id << ";";

	oss << "SELECT t.x, t.y FROM path_location pl "
		"JOIN two_d_location t ON pl.two_d_location_id = t.two_d_location_id "
		"WHERE pl.path_id = " << path_id << " "
		"ORDER BY pl.path_location_id;";

	int rc = sqlite3_prepare_v2(db, oss.str().c_str(), -1, &stmt, nullptr);

	if (rc != SQLITE_OK)
	{
		std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
		return path_points;
	}

	bool done = false;

	while (!done)
	{
		switch (sqlite3_step(stmt))
		{
		case SQLITE_ROW:
		{
			double x = sqlite3_column_double(stmt, 0);
			double y = sqlite3_column_double(stmt, 1);

			path_points.push_back(glm::vec2(x, y));

			row_count++;

			break;
		}
		case SQLITE_DONE:
		{
			done = true;
			break;
		}
		default:
		{
			done = true;
			cout << "Failure" << endl;
			break;
		}
		}
	}

	sqlite3_finalize(stmt);
	return path_points;
}





double get_path_animation_length(int path_id, sqlite3* (&db))
{
	size_t row_count = 0;

	double path_animation_length = 0;

	sqlite3_stmt* stmt;

	ostringstream oss;
	oss << "SELECT path_animation_length FROM path WHERE path_id = " << path_id << ";";

	int rc = sqlite3_prepare_v2(db, oss.str().c_str(), -1, &stmt, nullptr);

	if (rc != SQLITE_OK)
	{
		std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
		return 0.0;
	}

	bool done = false;

	while (!done)
	{
		switch (sqlite3_step(stmt))
		{
		case SQLITE_ROW:
		{
			path_animation_length = sqlite3_column_double(stmt, 0);

			row_count++;

			break;
		}
		case SQLITE_DONE:
		{
			done = true;
			break;
		}
		default:
		{
			done = true;
			cout << "Failure" << endl;
			break;
		}
		}
	}

	sqlite3_finalize(stmt);
	return path_animation_length;
}




vector<cannon> get_cannons(int enemy_id, sqlite3* (&db))
{
	size_t row_count = 0;

	vector<cannon> cannons;

	sqlite3_stmt* stmt;

	ostringstream oss;

	//	oss << "SELECT ec.enemy_id, ec.min_bullet_interval, ct.cannon_type_id, t.x, t.y FROM enemy_cannon ec JOIN cannon_type ct ON ec.cannon_type_id = ct.cannon_type_id JOIN two_d_location t ON ec.two_d_location_id = t.two_d_location_id WHERE enemy_id = " << enemy_id << ";";

	oss << "SELECT ec.enemy_id, ec.min_bullet_interval, ct.cannon_type_id, t.x, t.y "
		"FROM enemy_cannon ec "
		"JOIN cannon_type ct ON ec.cannon_type_id = ct.cannon_type_id "
		"JOIN two_d_location t ON ec.two_d_location_id = t.two_d_location_id "
		"WHERE ec.enemy_id = " << enemy_id << " "
		"ORDER BY ec.enemy_cannon_id;";


	int rc = sqlite3_prepare_v2(db, oss.str().c_str(), -1, &stmt, nullptr);

	if (rc != SQLITE_OK)
	{
		std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
		return cannons;
	}

	bool done = false;

	while (!done)
	{
		switch (sqlite3_step(stmt))
		{
		case SQLITE_ROW:
		{
			// int enemy_id = sqlite3_column_int(stmt, 0);

			cannon c;

			c.min_bullet_interval = sqlite3_column_double(stmt, 1);
			c.cannon_type = sqlite3_column_int(stmt, 2);
			c.x = sqlite3_column_double(stmt, 3);
			c.y = sqlite3_column_double(stmt, 4);

			cannons.push_back(c);

			row_count++;

			break;
		}
		case SQLITE_DONE:
		{
			done = true;
			break;
		}
		default:
		{
			done = true;
			cout << "Failure" << endl;
			break;
		}
		}
	}

	sqlite3_finalize(stmt);
	return cannons;
}

vector<int> get_power_ups(int enemy_id, sqlite3* (&db))
{
	vector<int> power_ups;

	sqlite3_stmt* stmt;

	ostringstream oss;
	oss << "SELECT power_up_id FROM enemy_power_up "
		"WHERE enemy_id = " << enemy_id << " "
		"ORDER BY enemy_power_up_id;";

	int rc = sqlite3_prepare_v2(db, oss.str().c_str(), -1, &stmt, nullptr);

	if (rc != SQLITE_OK)
	{
		std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
		return power_ups;
	}

	bool done = false;

	while (!done)
	{
		switch (sqlite3_step(stmt))
		{
		case SQLITE_ROW:
		{
			// DB stores power_up_id as 1-based; convert to 0-based in memory
			int pid = sqlite3_column_int(stmt, 0) - 1;
			power_ups.push_back(pid);
			break;
		}
		case SQLITE_DONE:
		{
			done = true;
			break;
		}
		default:
		{
			done = true;
			cout << "Failure" << endl;
			break;
		}
		}
	}

	sqlite3_finalize(stmt);
	return power_ups;
}

void retrieve_level_data(const string& db_name)
{
	enemy_ships.clear();

	size_t row_count = 0;

	sqlite3* db = 0;
	sqlite3_stmt* stmt = 0;
	string sql = "SELECT file_template_id, path_id, path_pixel_delay, max_health FROM enemy;";
	int rc = sqlite3_open(db_name.c_str(), &db);

	if (rc)
	{
		std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
		return;
	}

	// Bootstrap the schema so a brand-new or empty level file loads cleanly
	// instead of failing the SELECT below with "no such table: enemy".
	ensureDatabaseSchema(db);

	rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);

	if (rc != SQLITE_OK)
	{
		std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
		return;
	}

	bool done = false;

	while (!done)
	{
		switch (sqlite3_step(stmt))
		{
		case SQLITE_ROW:
		{
			// Note: file_template_id is 0-based (C++'s behaviour)
			int file_template_id = sqlite3_column_int(stmt, 0);

			// Note: path_id is 1-based (SQLite's default behaviour)
			int path_id = sqlite3_column_int(stmt, 1);

			int path_pixel_delay = sqlite3_column_int(stmt, 2);
			double max_health = sqlite3_column_double(stmt, 3);


			size_t enemy_template_index = file_template_id;
			enemy_ships.push_back(make_unique<enemy_ship>(enemy_templates[enemy_template_index]));
			enemy_ships.back()->template_idx = static_cast<int>(enemy_template_index);

			// Note: enemy_id is 1-based (SQLite's default behaviour)
			enemy_ships[enemy_ships.size() - 1]->cannons = get_cannons(static_cast<int>(enemy_ships.size()), db);

			enemy_ships[enemy_ships.size() - 1]->power_ups = get_power_ups(static_cast<int>(enemy_ships.size()), db);

			for (size_t i = 0; i < enemy_ships[enemy_ships.size() - 1]->cannons.size(); i++)
			{
				enemy_ships[enemy_ships.size() - 1]->cannons[i].x *= enemy_ships[enemy_ships.size() - 1]->width - 1;
				enemy_ships[enemy_ships.size() - 1]->cannons[i].y *= enemy_ships[enemy_ships.size() - 1]->height - 1;
				enemy_ships[enemy_ships.size() - 1]->cannons[i].cannon_type -= 1; // Switch from 1-based to 0-based
			}

			float half_w = enemy_ships[enemy_ships.size() - 1]->width / 2.0f;

			enemy_ships[enemy_ships.size() - 1]->path_points =
				get_path_points(path_id, db);

			for (size_t i = 0; i < enemy_ships[enemy_ships.size() - 1]->path_points.size(); i++)
			{
				enemy_ships[enemy_ships.size() - 1]->path_points[i].x *= SIM_WIDTH;
				enemy_ships[enemy_ships.size() - 1]->path_points[i].y *= SIM_HEIGHT;
			}

			enemy_ships[enemy_ships.size() - 1]->path_speeds =
				get_path_speeds(path_id, db);


			enemy_ships[enemy_ships.size() - 1]->path_animation_length =
				static_cast<float>(get_path_animation_length(path_id, db));


			glm::vec2 start_pos = get_spline_point(enemy_ships[enemy_ships.size() - 1]->path_points, 0.0f);
			enemy_ships[enemy_ships.size() - 1]->y = start_pos.y - enemy_ships[enemy_ships.size() - 1]->height * 0.5f;

			enemy_ships[enemy_ships.size() - 1]->health = enemy_ships[enemy_ships.size() - 1]->max_health = static_cast<float>(max_health);

			// Push the enemy further offscreen by the desired distance.
			// It will drift left at foreground_vel until it enters the screen,
			// at which point the spline path takes over.
			float desired_foreground_distance = static_cast<float>(path_pixel_delay);

			// The initial enemy x is the spawn knot's canonical position
			// (start_pos.x - half_w).  The path-point shift below re-bakes
			// the delay into the path so the spline's spawn knot ends up
			// at x + half_w + delay; adding the delay here as well would
			// double-apply it and desync the enemy from its own path.
			//
			// The initial location must not depend on the editor's scroll
			// state at save time, and path_pixel_delay must round-trip
			// unchanged through save/load.
			enemy_ships[enemy_ships.size() - 1]->x = start_pos.x - half_w;

			enemy_ships[enemy_ships.size() - 1]->path_pixel_delay = path_pixel_delay;

			// Fresh load: no spline-phase drift has been accumulated yet.
			enemy_ships[enemy_ships.size() - 1]->spline_phase_drift = 0.0f;

			// Shift path points by the pixel delay so the spline starts
			// where the enemy actually is.  The foreground drift during
			// gameplay will cancel this out by the time the enemy activates.
			for (size_t i = 0; i < enemy_ships[enemy_ships.size() - 1]->path_points.size(); i++)
				enemy_ships[enemy_ships.size() - 1]->path_points[i].x += desired_foreground_distance;

			enemy_ships[enemy_ships.size() - 1]->manually_update_data(
				enemy_templates[enemy_template_index].sprite_frames);
			enemy_ships[enemy_ships.size() - 1]->rebuildChunks();

			enemy_ships[enemy_ships.size() - 1]->set_velocity(enemy_ships[enemy_ships.size() - 1]->vel_x, enemy_ships[enemy_ships.size() - 1]->vel_y);





			row_count++;







			break;
		}
		case SQLITE_DONE:
		{
			done = true;
			break;
		}
		default:
		{
			done = true;
			cout << "Failure" << endl;
			break;
		}
		}
	}

	sqlite3_finalize(stmt);
	sqlite3_close(db);

	//if (!foreground_chunked.empty())
	//{
	//	g_loadTimeFgX = foreground_chunked[0].x;
	//}
	//else
	//{
	//	g_loadTimeFgX = 0.0f;
	//}
}



void load_media(const char* level_string)
{
	ms_music.setLooping(true);
	ms_music.setVolume(75.0f);
	ms_music.play();

	// Load protagonist texture -- scan for protagonist0.png, protagonist1.png, ...
	// The number of files must be odd (e.g. 3, 5, 7).
	{
		std::vector<std::string> proto_files;
		for (int i = 0; ; i++)
		{
			std::string path = "media/protagonist" + std::to_string(i) + ".png";
			if (!fs::exists(path)) break;
			proto_files.push_back(path);
		}

		if (proto_files.empty())
		{
			std::cout << "Warning: No protagonist sprite files found (expected media/protagonist0.png ...)" << std::endl;
			return;
		}
		if (proto_files.size() % 2 == 0)
		{
			std::cout << "Warning: Protagonist has " << proto_files.size()
				<< " frames but needs an odd number. Dropping the last frame." << std::endl;
			proto_files.pop_back();
		}

		protagonist.tex = loadTextureFromFile_NSprite(proto_files,
			&protagonist.width, &protagonist.height, protagonist);
	}
	if (protagonist.tex == 0)
	{
		std::cout << "Warning: Could not load protagonist sprite" << std::endl;
		return;
	}

	protagonist.x = 200;
	protagonist.y = 300;

	bullet_template.tex = loadTextureFromFile("media/bullet.png", &bullet_template.width, &bullet_template.height, bullet_template.to_present_data);
	if (bullet_template.tex == 0)
	{
		std::cout << "Warning: Could not load bullet_template sprite" << std::endl;
		return;
	}

	game_over_banner.tex = loadTextureFromFile("media/game_over.png", &game_over_banner.width, &game_over_banner.height, game_over_banner.to_present_data);
	if (game_over_banner.tex == 0)
	{
		std::cout << "Warning: Could not load game_over_banner sprite" << std::endl;
		return;
	}

	// Power-up textures. Warn (don't return) so the level still loads if art
	// assets are missing. Rename these paths to match your actual files.
	power_up_template_sinusoidal.tex = loadTextureFromFile("media/sinusoidal_fire.png",
		&power_up_template_sinusoidal.width, &power_up_template_sinusoidal.height,
		power_up_template_sinusoidal.to_present_data);
	if (power_up_template_sinusoidal.tex == 0)
		std::cout << "Warning: Could not load sinusoidal_fire.png" << std::endl;

	power_up_template_x3.tex = loadTextureFromFile("media/x3_fire.png",
		&power_up_template_x3.width, &power_up_template_x3.height,
		power_up_template_x3.to_present_data);
	if (power_up_template_x3.tex == 0)
		std::cout << "Warning: Could not load x3_fire.png" << std::endl;

	power_up_template_x5.tex = loadTextureFromFile("media/x5_fire.png",
		&power_up_template_x5.width, &power_up_template_x5.height,
		power_up_template_x5.to_present_data);
	if (power_up_template_x5.tex == 0)
		std::cout << "Warning: Could not load x5_fire.png" << std::endl;

	string affix = "media/";
	affix += level_string;
	affix += "/";

	string s = affix + "background.png";

	background.tex = loadTextureFromFile(s.c_str(), &background.width, &background.height, background.to_present_data, true);
	if (background.tex == 0)
	{
		std::cout << "Warning: Could not load background sprite" << std::endl;
		return;
	}

	s = affix + "background_lit.png";

	background_lit.tex = loadTextureFromFile(s.c_str(), &background_lit.width, &background_lit.height, background_lit.to_present_data, true);
	if (background_lit.tex == 0)
	{
		std::cout << "Warning: Could not load background_lit sprite" << std::endl;
		return;
	}

	s = affix + "foreground.png";
	if (!chunkForegroundTexture(s.c_str()))
	{
		std::cout << "Warning: Could not chunk foreground sprite" << std::endl;
		return;
	}

	s = affix + "foreground_lit.png";
	if (!chunkForegroundTexture(s.c_str(), foreground_lit_chunked))
	{
		std::cout << "Warning: Could not chunk foreground_lit sprite" << std::endl;
		return;
	}



	// Dynamically load all enemy templates from the level's media directory.
	// New naming convention: enemy_<enemyIdx>_<frameIdx>.png
	// e.g. enemy_0_0.png, enemy_0_1.png, ..., enemy_0_4.png (5 frames for enemy 0)
	//      enemy_1_0.png, enemy_1_1.png, ...                  (frames for enemy 1)
	// Frame count per enemy must be odd.
	{
		// First, discover which enemy indices exist by scanning for enemy_*_0.png
		std::vector<int> enemy_indices;

		string s = "media/";
		s += level_string;

		for (const auto& entry : fs::directory_iterator(s))
		{
			if (!entry.is_regular_file()) continue;

			std::string filename = entry.path().filename().string();

			// Match pattern: enemy_<N>_0.png  (the first frame of each enemy)
			if (filename.rfind("enemy_", 0) == 0 && filename.find("_0.png") != std::string::npos)
			{
				// Extract the enemy index between the two underscores
				// "enemy_X_0.png" -> X
				size_t first_us = filename.find('_');       // after "enemy"
				size_t second_us = filename.find('_', first_us + 1); // before "0.png"
				if (first_us != std::string::npos && second_us != std::string::npos)
				{
					std::string idx_str = filename.substr(first_us + 1, second_us - first_us - 1);
					try { enemy_indices.push_back(std::stoi(idx_str)); }
					catch (...) { /* not a number, skip */ }
				}
			}
		}

		std::sort(enemy_indices.begin(), enemy_indices.end());

		// Load each enemy template
		for (int enemyIdx : enemy_indices)
		{
			// Collect all frame files for this enemy index
			std::vector<std::string> frame_files;
			for (int f = 0; ; f++)
			{
				std::string path = affix + "enemy_" + std::to_string(enemyIdx) + "_" + std::to_string(f) + ".png";
				if (!fs::exists(path)) break;
				frame_files.push_back(path);
			}

			if (frame_files.empty())
			{
				std::cout << "Warning: No frames found for enemy_" << enemyIdx << std::endl;
				continue;
			}

			if (frame_files.size() % 2 == 0)
			{
				std::cout << "Warning: enemy_" << enemyIdx << " has " << frame_files.size()
					<< " frames but needs an odd number. Dropping the last frame." << std::endl;
				frame_files.pop_back();
			}

			enemy_ship new_template;
			new_template.tex = loadTextureFromFile_NSprite(
				frame_files,
				&new_template.width, &new_template.height,
				new_template);

			if (new_template.tex == 0)
			{
				std::cout << "Warning: Could not load enemy_" << enemyIdx << " sprite" << std::endl;
				continue;
			}

			enemy_templates.push_back(std::move(new_template));
			std::cout << "Loaded enemy template: enemy_" << enemyIdx
				<< " (" << frame_files.size() << " frames)" << std::endl;
		}

		if (enemy_templates.empty())
		{
			std::cout << "Warning: No enemy templates found in media directory" << std::endl;
			return;
		}



		retrieve_level_data("level1.db");

		g_loadTimeFgX = foreground_chunked.empty()
			? 0.0f
			: foreground_chunked[0].x;


	}
}


void reset_game()
{
	// ---- Clear all bullets ----
	ally_bullets.clear();
	enemy_bullets.clear();
	power_ups_alive.clear();

	// ---- Clear and reload enemy ships from the database ----
	// retrieve_level_data already calls enemy_ships.clear() internally
	retrieve_level_data("level1.db");

	// ---- Reset protagonist ----
	// Delete old texture to avoid leak, then reload from disk to undo blackening
	if (protagonist.tex) {
		glDeleteTextures(1, &protagonist.tex);
		protagonist.tex = 0;
	}
	// Reload protagonist frames: protagonist0.png, protagonist1.png, ...
	{
		std::vector<std::string> proto_files;
		for (int i = 0; ; i++)
		{
			std::string path = "media/protagonist" + std::to_string(i) + ".png";
			if (!fs::exists(path)) break;
			proto_files.push_back(path);
		}
		if (proto_files.size() % 2 == 0 && !proto_files.empty())
			proto_files.pop_back();

		protagonist.tex = loadTextureFromFile_NSprite(proto_files,
			&protagonist.width, &protagonist.height, protagonist);
	}
	protagonist.x = 200;
	protagonist.y = 300;
	protagonist.vel_x = 0;
	protagonist.vel_y = 0;
	protagonist.old_x = protagonist.x;
	protagonist.old_y = protagonist.y;
	protagonist.health = 1000.0f;
	protagonist.max_health = 1000.0f;
	protagonist.to_be_culled = false;
	protagonist.under_fire = false;
	protagonist.last_time_collided = 0;
	protagonist.blackening_age_map.clear();
	protagonist.state = protagonist.rest_state_index();
	protagonist.update_tex();

	// ---- Re-chunk foreground from disk (resets positions + blackening) ----
	// Delete old tile textures to avoid leak
	for (size_t i = 0; i < foreground_chunked.size(); i++) {
		if (foreground_chunked[i].tex) {
			glDeleteTextures(1, &foreground_chunked[i].tex);
		}
	}
	chunkForegroundTexture("media/level1/foreground.png");

	// Same cleanup + reload for the lit overlay
	for (size_t i = 0; i < foreground_lit_chunked.size(); i++) {
		if (foreground_lit_chunked[i].tex) {
			glDeleteTextures(1, &foreground_lit_chunked[i].tex);
		}
	}
	chunkForegroundTexture("media/level1/foreground_lit.png", foreground_lit_chunked);

	// ---- Reset timing ----
	GLOBAL_TIME = 0;
	lastDamageTime = -1;
	lastBulletTime = std::chrono::high_resolution_clock::now();

	// ---- Reset wave events ----
	for (int i = 0; i < MAX_WAVE_SOURCES; i++) {
		waveEvents[i] = WaveEvent();
	}
	nextWaveSlot = 0;

	// ---- Reset input state ----
	spacePressed = false;
	upKeyPressed = false;
	downKeyPressed = false;
	leftKeyPressed = false;
	rightKeyPressed = false;

	// ---- Clear fluid simulation FBOs ----
	for (int i = 0; i < 2; i++) {
		glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[i]);
		glClear(GL_COLOR_BUFFER_BIT);
		glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[i]);
		glClear(GL_COLOR_BUFFER_BIT);
		glBindFramebuffer(GL_FRAMEBUFFER, densityFBO[i]);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	// ---- Clear obstacles ----
	glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	std::cout << "=== GAME RESET ===" << std::endl;
}


bool editorHandleKey(unsigned char key, int /*mx*/, int /*my*/)
{
	if (key == '\t')
	{
		g_editorMode = !g_editorMode;
		g_selectedPoint = -1;
		g_draggingPoint = false;

		if (g_editorMode)
		{
			editorResetUndoHistory();
			// Snap all enemies to their rest state so cannon positions
			// displayed in the editor match the frame they were defined on
			for (auto& es : enemy_ships)
			{
				es->state = es->rest_state_index();
				es->update_tex();
			}
		}

		std::cout << "[Editor] " << (g_editorMode ? "ON" : "OFF") << "\n";
		return true;
	}

	if (!g_editorMode) return false;

	enemy_ship* e = editorSelected();

	switch (key)
	{
	case 'e': case 'E':
		if (e && !enemy_templates.empty())
		{
			editorPushUndo();
			int cur = editorFindTemplateIdx(e);
			int next = (cur + 1) % (int)enemy_templates.size();
			editorApplyTemplate(e, next);
			std::cout << "[Editor] Template -> " << next
				<< "  (" << next + 1 << "/" << enemy_templates.size() << ")\n";
		}
		return true;

	case '[':
		if (!enemy_ships.empty())
			g_selectedEnemy = (g_selectedEnemy - 1 + (int)enemy_ships.size()) % (int)enemy_ships.size();
		g_selectedPoint = -1;
		g_selectedSpeedKnot = -1;
		g_selectedCannon = -1;
		g_selectedPowerUp = -1;
		return true;

	case ']':
		if (!enemy_ships.empty())
			g_selectedEnemy = (g_selectedEnemy + 1) % (int)enemy_ships.size();
		g_selectedPoint = -1;
		g_selectedSpeedKnot = -1;
		g_selectedCannon = -1;
		g_selectedPowerUp = -1;
		return true;

	case '{':
		if (e && !e->cannons.empty())
		{
			if (g_selectedCannon < 0)
				g_selectedCannon = (int)e->cannons.size() - 1;
			else
				g_selectedCannon = (g_selectedCannon - 1 + (int)e->cannons.size()) % (int)e->cannons.size();
			std::cout << "[Editor] Selected cannon " << g_selectedCannon
				<< " / " << (int)e->cannons.size() - 1
				<< "  type=" << e->cannons[g_selectedCannon].cannon_type
				<< "  interval=" << e->cannons[g_selectedCannon].min_bullet_interval << "s\n";
		}
		return true;

	case '}':
		if (e && !e->cannons.empty())
		{
			g_selectedCannon = (std::max(0, g_selectedCannon) + 1) % (int)e->cannons.size();
			std::cout << "[Editor] Selected cannon " << g_selectedCannon
				<< " / " << (int)e->cannons.size() - 1
				<< "  type=" << e->cannons[g_selectedCannon].cannon_type
				<< "  interval=" << e->cannons[g_selectedCannon].min_bullet_interval << "s\n";
		}
		return true;

	case 'n': case 'N':
		if (!enemy_templates.empty())
		{
			editorPushUndo();

			// Snap every existing enemy to its rightmost path knot before
			// spawning the new one — but skip any that have already begun
			// spline interpolation (path_t > 0), so we don't yank them
			// backwards along their path.
			for (auto& es : enemy_ships)
			{
				if (es->path_points.empty()) continue;
				if (es->path_t > 0.0f) continue; // already moving along spline

				// path_points.front() is the right-side entry knot.
				es->x = es->path_points.front().x - es->width * 0.5f;
				es->y = es->path_points.front().y - es->height * 0.5f;
			}

			int tIdx = 0;//g_spawnTemplateIdx % (int)enemy_templates.size();
			enemy_ships.push_back(std::make_unique<enemy_ship>(enemy_templates[tIdx]));
			enemy_ship* ne = enemy_ships.back().get();
			ne->template_idx = tIdx;
			ne->x = SIM_WIDTH * 0.5f - ne->width * 0.5f;
			ne->y = SIM_HEIGHT * 0.5f - ne->height * 0.5f;
			ne->health = ne->max_health = 100.0f;
			ne->path_animation_length = 5.0f;
			ne->path_points.push_back(glm::vec2(SIM_WIDTH + ne->width / 2.0, SIM_HEIGHT * 0.5f));
			ne->path_points.push_back(glm::vec2(-(float)ne->width / 2.0, SIM_HEIGHT * 0.5f));
			ne->path_speeds.push_back(1.0f);
			ne->path_speeds.push_back(1.0f);

			float actual_duration = calculate_actual_path_duration(
				ne->path_points, ne->path_speeds, ne->path_animation_length);

			ne->path_scroll_rate = -(ne->width * 0.5f) / actual_duration;

			float foreground_scrolled = -foreground_vel * GLOBAL_TIME;
			ne->path_pixel_delay = (int)(ne->path_points.back().x + ne->width * 0.5f + foreground_scrolled);

			ne->to_be_culled = false;

			ne->manually_update_data(enemy_templates[tIdx].sprite_frames);
			ne->rebuildChunks();

			g_selectedEnemy = (int)enemy_ships.size() - 1;
			g_spawnTemplateIdx++;
			std::cout << "[Editor] Spawned enemy " << g_selectedEnemy << "\n";
		}
		return true;

	case 'd': case 'D':
		if (e)
		{
			editorPushUndo();
			enemy_ships.erase(enemy_ships.begin() + g_selectedEnemy);
			g_selectedEnemy = std::max(0, g_selectedEnemy - 1);
			g_selectedPoint = -1;
			std::cout << "[Editor] Deleted enemy\n";
		}
		return true;

	case 'c': case 'C':
		if (e)
		{
			editorPushUndo();
			cannon c;
			c.x = std::max(0.0, std::min((double)(mouseX - e->x), (double)(e->width - 1)));
			c.y = std::max(0.0, std::min((double)(mouseY - e->y), (double)(e->height - 1)));
			c.cannon_type = CANNON_TYPE_LEFT;
			c.min_bullet_interval = 2.0;
			e->cannons.push_back(c);
			g_selectedCannon = (int)e->cannons.size() - 1;
			std::cout << "[Editor] Added cannon " << g_selectedCannon
				<< " at local (" << c.x << ", " << c.y << ")\n";
		}
		return true;

	case 'v': case 'V':
		if (e && !e->cannons.empty())
		{
			editorPushUndo();
			int removeIdx = (g_selectedCannon >= 0 && g_selectedCannon < (int)e->cannons.size())
				? g_selectedCannon : (int)e->cannons.size() - 1;
			e->cannons.erase(e->cannons.begin() + removeIdx);
			// Clamp selection to the new size
			if (e->cannons.empty())
				g_selectedCannon = -1;
			else
				g_selectedCannon = std::min(removeIdx, (int)e->cannons.size() - 1);
			std::cout << "[Editor] Removed cannon " << removeIdx << "\n";
		}
		return true;

	case 't': case 'T':
		if (e && !e->cannons.empty())
		{
			editorPushUndo();
			int idx = (g_selectedCannon >= 0 && g_selectedCannon < (int)e->cannons.size())
				? g_selectedCannon : (int)e->cannons.size() - 1;
			cannon& lc = e->cannons[idx];
			lc.cannon_type = (lc.cannon_type + 1) % 4;
			const char* names[] = { "LEFT", "UP_DOWN", "TRACKING", "CIRCULAR" };
			std::cout << "[Editor] Cannon " << idx << " type -> " << names[lc.cannon_type] << "\n";
		}
		return true;

	case ',':
		if (e && !e->cannons.empty())
		{
			editorPushUndo();
			int idx = (g_selectedCannon >= 0 && g_selectedCannon < (int)e->cannons.size())
				? g_selectedCannon : (int)e->cannons.size() - 1;
			e->cannons[idx].min_bullet_interval =
				std::max(0.1, e->cannons[idx].min_bullet_interval - 0.1);
			std::cout << "[Editor] Cannon " << idx << " fire interval: " << e->cannons[idx].min_bullet_interval << "s\n";
		}
		return true;

	case '.':
		if (e && !e->cannons.empty())
		{
			editorPushUndo();
			int idx = (g_selectedCannon >= 0 && g_selectedCannon < (int)e->cannons.size())
				? g_selectedCannon : (int)e->cannons.size() - 1;
			e->cannons[idx].min_bullet_interval += 0.1;
			std::cout << "[Editor] Cannon " << idx << " fire interval: " << e->cannons[idx].min_bullet_interval << "s\n";
		}
		return true;

	case 'b': case 'B':
		// Add a new power-up to the selected enemy (default: sinusoidal)
		if (e)
		{
			editorPushUndo();
			e->power_ups.push_back(POWER_UP_TYPE_SINUSOIDAL);
			g_selectedPowerUp = (int)e->power_ups.size() - 1;
			const char* names[] = { "SINUSOIDAL", "X3", "X5" };
			std::cout << "[Editor] Added power-up " << g_selectedPowerUp
				<< " type=" << names[e->power_ups[g_selectedPowerUp]] << "\n";
		}
		return true;

	case ':':
		// Select previous power-up on the selected enemy
		if (e && !e->power_ups.empty())
		{
			if (g_selectedPowerUp < 0)
				g_selectedPowerUp = (int)e->power_ups.size() - 1;
			else
				g_selectedPowerUp = (g_selectedPowerUp - 1 + (int)e->power_ups.size()) % (int)e->power_ups.size();
			const char* names[] = { "SINUSOIDAL", "X3", "X5" };
			std::cout << "[Editor] Selected power-up " << g_selectedPowerUp
				<< " / " << (int)e->power_ups.size() - 1
				<< "  type=" << names[e->power_ups[g_selectedPowerUp]] << "\n";
		}
		return true;

	case '\'':
		// Select next power-up on the selected enemy
		if (e && !e->power_ups.empty())
		{
			g_selectedPowerUp = (std::max(0, g_selectedPowerUp) + 1) % (int)e->power_ups.size();
			const char* names[] = { "SINUSOIDAL", "X3", "X5" };
			std::cout << "[Editor] Selected power-up " << g_selectedPowerUp
				<< " / " << (int)e->power_ups.size() - 1
				<< "  type=" << names[e->power_ups[g_selectedPowerUp]] << "\n";
		}
		return true;

	case 'f': case 'F':
		// Delete the selected power-up (or last if none selected)
		if (e && !e->power_ups.empty())
		{
			editorPushUndo();
			int removeIdx = (g_selectedPowerUp >= 0 && g_selectedPowerUp < (int)e->power_ups.size())
				? g_selectedPowerUp : (int)e->power_ups.size() - 1;
			e->power_ups.erase(e->power_ups.begin() + removeIdx);
			if (e->power_ups.empty())
				g_selectedPowerUp = -1;
			else
				g_selectedPowerUp = std::min(removeIdx, (int)e->power_ups.size() - 1);
			std::cout << "[Editor] Removed power-up " << removeIdx << "\n";
		}
		return true;

	case 'y': case 'Y':
		// Cycle the type of the selected power-up
		// NOTE: case 25 below handles Ctrl+Y for redo; plain 'y' is 121, no collision.
		if (e && !e->power_ups.empty())
		{
			editorPushUndo();
			int idx = (g_selectedPowerUp >= 0 && g_selectedPowerUp < (int)e->power_ups.size())
				? g_selectedPowerUp : (int)e->power_ups.size() - 1;
			e->power_ups[idx] = (e->power_ups[idx] + 1) % NUM_POWER_UP_TYPES;
			const char* names[] = { "SINUSOIDAL", "X3", "X5" };
			std::cout << "[Editor] Power-up " << idx << " type -> " << names[e->power_ups[idx]] << "\n";
		}
		return true;


	case 'a':
	case 'A':
		editorSaveToDatabase("level1.db");
		reset_game();
		g_editorMode = false;

		break;



	case '-': case '_':
		// Decrease selected speed knot value by 0.1 (min 0.1)
		if (e && g_selectedSpeedKnot >= 0 && g_selectedSpeedKnot < (int)e->path_speeds.size())
		{
			editorPushUndo();
			e->path_speeds[g_selectedSpeedKnot] =
				std::max(0.1f, e->path_speeds[g_selectedSpeedKnot] - 0.1f);
			std::cout << "[Editor] Speed knot " << g_selectedSpeedKnot
				<< " -> " << e->path_speeds[g_selectedSpeedKnot] << "\n";
		}
		return true;

	case '=': case '+':
		// Increase selected speed knot value by 0.1
		if (e && g_selectedSpeedKnot >= 0 && g_selectedSpeedKnot < (int)e->path_speeds.size())
		{
			editorPushUndo();
			e->path_speeds[g_selectedSpeedKnot] += 0.1f;
			std::cout << "[Editor] Speed knot " << g_selectedSpeedKnot
				<< " -> " << e->path_speeds[g_selectedSpeedKnot] << "\n";
		}
		return true;

	case 'h':
		// Decrease selected enemy's max_health by 100 (min 100)
		if (e)
		{
			editorPushUndo();
			e->max_health = std::max(100.0f, e->max_health - 100.0f);
			e->health = std::min(e->health, e->max_health);
			std::cout << "[Editor] Enemy " << g_selectedEnemy
				<< " max_health -> " << e->max_health << "\n";
		}
		return true;

	case 'H':
		// Increase selected enemy's max_health by 100
		if (e)
		{
			editorPushUndo();
			e->max_health += 100.0f;
			e->health = e->max_health;
			std::cout << "[Editor] Enemy " << g_selectedEnemy
				<< " max_health -> " << e->max_health << "\n";
		}
		return true;

	case 'j':
		// Decrease selected enemy's path_animation_length by 1.0 (min 0.1)
		if (e)
		{
			editorPushUndo();
			e->path_animation_length = std::max(0.1f, e->path_animation_length - 1.0f);
			if (e->path_animation_length > 0.0f && e->path_points.size() >= 2)
			{
				float actual_duration = calculate_actual_path_duration(
					e->path_points, e->path_speeds, e->path_animation_length);
				e->path_scroll_rate = -(e->width * 0.5f) / actual_duration;
			}
			std::cout << "[Editor] Enemy " << g_selectedEnemy
				<< " path_animation_length -> " << e->path_animation_length << "s\n";
		}
		return true;

	case 'J':
		// Increase selected enemy's path_animation_length by 1.0
		if (e)
		{
			editorPushUndo();
			e->path_animation_length += 1.0f;
			if (e->path_animation_length > 0.0f && e->path_points.size() >= 2)
			{
				float actual_duration = calculate_actual_path_duration(
					e->path_points, e->path_speeds, e->path_animation_length);
				e->path_scroll_rate = -(e->width * 0.5f) / actual_duration;
			}
			std::cout << "[Editor] Enemy " << g_selectedEnemy
				<< " path_animation_length -> " << e->path_animation_length << "s\n";
		}
		return true;

	case 26: // Ctrl+Z — Undo
		editorUndo();
		return true;

	case 25: // Ctrl+Y — Redo
		editorRedo();
		return true;

	case 's':
		draw_time_lines = !draw_time_lines;

		return true;

	case 3: // Ctrl+C / Ctrl+Shift+C
		if (e)
		{
			bool shift = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
			if (shift)
			{
				// Ctrl+Shift+C — copy entire enemy
				g_enemy_clipboard.template_idx = e->template_idx;
				g_enemy_clipboard.path_points = e->path_points;
				g_enemy_clipboard.path_speeds = e->path_speeds;
				g_enemy_clipboard.cannons = e->cannons;
				g_enemy_clipboard.power_ups = e->power_ups;
				g_enemy_clipboard.path_animation_length = e->path_animation_length;
				g_enemy_clipboard.health = e->health;
				g_enemy_clipboard.max_health = e->max_health;
				g_enemy_clipboard.path_pixel_delay = e->path_pixel_delay;
				g_enemy_clipboard_has_data = true;
				std::cout << "[Editor] Copied enemy " << g_selectedEnemy
					<< " (template " << e->template_idx
					<< ", " << e->path_points.size() << " points, "
					<< e->cannons.size() << " cannons)\n";
			}
			else
			{
				// Ctrl+C — copy path data only
				g_clipboard_path_points = e->path_points;
				g_clipboard_path_speeds = e->path_speeds;
				g_clipboard_has_data = true;
				std::cout << "[Editor] Copied path data from enemy " << g_selectedEnemy
					<< "  (" << g_clipboard_path_points.size() << " points, "
					<< g_clipboard_path_speeds.size() << " speed knots)\n";
			}
		}
		return true;

	case 22: // Ctrl+V / Ctrl+Shift+V
	{
		bool shift = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
		if (shift)
		{
			// Ctrl+Shift+V — paste entire enemy
			if (!g_enemy_clipboard_has_data)
			{
				std::cout << "[Editor] Enemy clipboard is empty — use Ctrl+Shift+C on an enemy first\n";
				return true;
			}
			if (enemy_templates.empty())
			{
				std::cout << "[Editor] No enemy templates loaded\n";
				return true;
			}

			editorPushUndo();

			// If no enemy is selected (list is empty), create one first
			if (!e)
			{
				int tIdx = g_enemy_clipboard.template_idx;
				if (tIdx < 0 || tIdx >= (int)enemy_templates.size())
					tIdx = 0;

				enemy_ships.push_back(std::make_unique<enemy_ship>(enemy_templates[tIdx]));
				enemy_ship* ne = enemy_ships.back().get();
				ne->x = SIM_WIDTH * 0.5f - ne->width * 0.5f;
				ne->y = SIM_HEIGHT * 0.5f - ne->height * 0.5f;
				ne->to_be_culled = false;
				ne->manually_update_data(enemy_templates[tIdx].sprite_frames);
				ne->rebuildChunks();

				g_selectedEnemy = (int)enemy_ships.size() - 1;
				e = enemy_ships.back().get();
				std::cout << "[Editor] Created new enemy " << g_selectedEnemy << " for paste\n";
			}

			// Apply the correct template from the clipboard
			int tIdx = g_enemy_clipboard.template_idx;
			if (tIdx >= 0 && tIdx < (int)enemy_templates.size())
				editorApplyTemplate(e, tIdx);

			// Paste all saved properties
			e->path_points = g_enemy_clipboard.path_points;
			e->path_speeds = g_enemy_clipboard.path_speeds;
			e->cannons = g_enemy_clipboard.cannons;
			e->power_ups = g_enemy_clipboard.power_ups;
			e->path_animation_length = g_enemy_clipboard.path_animation_length;
			e->health = g_enemy_clipboard.health;
			e->max_health = g_enemy_clipboard.max_health;
			e->path_pixel_delay = g_enemy_clipboard.path_pixel_delay;

			g_selectedPoint = -1;
			g_selectedSpeedKnot = -1;
			g_selectedCannon = -1;
			g_selectedPowerUp = -1;
			g_draggingPoint = false;

			// Recalculate scroll rate
			if (e->path_animation_length > 0.0f)
			{
				float actual_duration = calculate_actual_path_duration(
					e->path_points, e->path_speeds, e->path_animation_length);
				e->path_scroll_rate = -(e->width * 0.5f) / actual_duration;
			}

			std::cout << "[Editor] Pasted enemy onto slot " << g_selectedEnemy
				<< " (template " << tIdx
				<< ", " << e->path_points.size() << " points, "
				<< e->cannons.size() << " cannons)\n";
		}
		else
		{
			// Ctrl+V — paste path data only
			if (e && g_clipboard_has_data)
			{
				editorPushUndo();
				e->path_points = g_clipboard_path_points;
				e->path_speeds = g_clipboard_path_speeds;
				g_selectedPoint = -1;
				g_selectedSpeedKnot = -1;
				g_draggingPoint = false;

				// Recalculate the scroll rate so the pasted path animates correctly
				if (e->path_animation_length > 0.0f)
				{
					float actual_duration = calculate_actual_path_duration(
						e->path_points, e->path_speeds, e->path_animation_length);
					e->path_scroll_rate = -(e->width * 0.5f) / actual_duration;
				}

				std::cout << "[Editor] Pasted path data onto enemy " << g_selectedEnemy
					<< "  (" << e->path_points.size() << " points, "
					<< e->path_speeds.size() << " speed knots)\n";
			}
			else if (!g_clipboard_has_data)
			{
				std::cout << "[Editor] Clipboard is empty — use Ctrl+C on an enemy first\n";
			}
		}
		return true;
	}

	default:
		break;
	}

	return false;
}

// ---- Mouse button handler (returns true if editor consumed the event) -------

bool editorHandleMouse(int button, int state, int mx, int my)
{
	if (!g_editorMode) return false;

	enemy_ship* e = editorSelected();
	if (!e) return false;

	bool shift = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;
	bool ctrl = (glutGetModifiers() & GLUT_ACTIVE_CTRL) != 0;

	// Ctrl+LMB: select the enemy whose bounding box is under the cursor
	if (ctrl && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		// Scale mouse coords from window space to sim space
		float simMx = (float)mx * SIM_WIDTH / (float)windowWidth;
		float simMy = (float)my * SIM_HEIGHT / (float)windowHeight;

		// Search back-to-front so the visually topmost enemy wins
		for (int i = (int)enemy_ships.size() - 1; i >= 0; --i)
		{
			const enemy_ship& es = *enemy_ships[i];
			if (es.to_be_culled) continue;

			if (simMx >= es.x && simMx <= es.x + es.width &&
				simMy >= es.y && simMy <= es.y + es.height)
			{
				g_selectedEnemy = i;
				g_selectedPoint = -1;
				g_selectedSpeedKnot = -1;
				g_draggingPoint = false;
				std::cout << "[Editor] Ctrl+Click selected enemy " << i << "\n";
				return true;
			}
		}
		// Nothing hit — leave selection unchanged
		std::cout << "[Editor] Ctrl+Click: no enemy under cursor\n";
		return true;
	}

	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			if (shift)
			{
				// Shift+LMB -> add speed knot
				editorPushUndo();
				e->path_speeds.push_back(1.f);
				std::cout << "[Editor] Added speed knot (val=1.0)\n";
			}
			else
			{
				int idx = editorFindNearestPoint(*e, (float)mx, (float)my, 20.f);
				int sIdx = editorFindNearestSpeedKnot(*e, (float)mx, (float)my, 25.f);

				// If both a path-point knot and a speed knot are under the cursor,
				// toggle to the other type when the clicked one is already selected.
				if (idx >= 0 && sIdx >= 0)
				{
					if (idx == g_selectedPoint)
					{
						// Path point already selected -> switch to speed knot
						g_selectedSpeedKnot = sIdx;
						g_selectedPoint = -1;
						g_draggingPoint = false;
						std::cout << "[Editor] Toggled to speed knot " << sIdx
							<< " (val=" << e->path_speeds[sIdx] << ")  "
							<< "Use scroll wheel or -/= to adjust\n";
					}
					else if (sIdx == g_selectedSpeedKnot)
					{
						// Speed knot already selected -> switch to path point
						g_selectedPoint = idx;
						g_selectedSpeedKnot = -1;
						g_draggingPoint = true;
						g_dragUndoPushed = false;
						std::cout << "[Editor] Toggled to path point " << idx << "\n";
					}
					else
					{
						// Neither is selected yet: prefer the path-point knot
						g_selectedPoint = idx;
						g_selectedSpeedKnot = -1;
						g_draggingPoint = true;
						g_dragUndoPushed = false;
					}
				}
				else if (idx >= 0)
				{
					g_selectedPoint = idx;
					g_selectedSpeedKnot = -1;
					g_draggingPoint = true;
					g_dragUndoPushed = false;
				}
				else if (sIdx >= 0)
				{
					g_selectedSpeedKnot = sIdx;
					g_selectedPoint = -1;
					std::cout << "[Editor] Selected speed knot " << sIdx
						<< " (val=" << e->path_speeds[sIdx] << ")  "
						<< "Use scroll wheel or -/= to adjust\n";
				}
				else
				{
					g_selectedSpeedKnot = -1;
					// Insert new control point in the nearest segment
					editorPushUndo();
					glm::vec2 np((float)mx, (float)my);
					int insertAfter = 0;
					float bestDist = 1e30f;
					for (size_t i = 0; i + 1 < e->path_points.size(); ++i)
					{
						glm::vec2 mid = (e->path_points[i] + e->path_points[i + 1]) * 0.5f;
						float dx = mid.x - np.x, dy = mid.y - np.y;
						float d = dx * dx + dy * dy;
						if (d < bestDist) { bestDist = d; insertAfter = (int)i; }
					}
					e->path_points.insert(e->path_points.begin() + insertAfter + 1, np);
					g_selectedPoint = insertAfter + 1;
					g_draggingPoint = true;
					g_dragUndoPushed = true; // insert already pushed undo
					std::cout << "[Editor] Inserted path point at (" << mx << ", " << my << ")\n";
				}
			}
		}
		else  // GLUT_UP
		{
			g_draggingPoint = false;
		}
		return true;
	}

	// Scroll wheel: adjust selected speed knot value
	if ((button == 3 || button == 4) && state == GLUT_DOWN)
	{
		if (g_selectedSpeedKnot >= 0 && g_selectedSpeedKnot < (int)e->path_speeds.size())
		{
			editorPushUndo();
			float delta = (button == 3) ? 0.1f : -0.1f;
			e->path_speeds[g_selectedSpeedKnot] =
				std::max(0.1f, e->path_speeds[g_selectedSpeedKnot] + delta);

			if (e->path_speeds[g_selectedSpeedKnot] > 1.0f)
				e->path_speeds[g_selectedSpeedKnot] = 1.0f;

			std::cout << "[Editor] Speed knot " << g_selectedSpeedKnot
				<< " -> " << e->path_speeds[g_selectedSpeedKnot] << "\n";
			return true;
		}
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		if (shift)
		{
			if (g_selectedSpeedKnot >= 0 && g_selectedSpeedKnot < (int)e->path_speeds.size())
			{
				editorPushUndo();
				std::cout << "[Editor] Removed selected speed knot " << g_selectedSpeedKnot
					<< " (val=" << e->path_speeds[g_selectedSpeedKnot] << ")\n";
				e->path_speeds.erase(e->path_speeds.begin() + g_selectedSpeedKnot);
				g_selectedSpeedKnot = -1;
			}
			else if (!e->path_speeds.empty())
			{
				editorPushUndo();
				e->path_speeds.pop_back();
				std::cout << "[Editor] Removed last speed knot\n";
			}
		}
		else
		{
			int idx = editorFindNearestPoint(*e, (float)mx, (float)my, 20.f);
			if (idx >= 0 && e->path_points.size() > 2)
			{
				editorPushUndo();
				e->path_points.erase(e->path_points.begin() + idx);
				if (g_selectedPoint >= (int)e->path_points.size())
					g_selectedPoint = (int)e->path_points.size() - 1;
				std::cout << "[Editor] Deleted path point " << idx << "\n";
			}
		}
		return true;
	}

	return false;
}

// ---- Motion handler (returns true if editor consumed the event) -------------

bool editorHandleMotion(int mx, int my)
{
	if (!g_editorMode || !g_draggingPoint) return false;

	enemy_ship* e = editorSelected();
	if (!e) return false;

	if (g_selectedPoint >= 0 && g_selectedPoint < (int)e->path_points.size())
	{
		if (!g_dragUndoPushed)
		{
			editorPushUndo();
			g_dragUndoPushed = true;
		}
		bool isEndpoint = (g_selectedPoint == 0 ||
			g_selectedPoint == (int)e->path_points.size() - 1);
		// Endpoints: X is fixed (must enter/exit off-screen); only Y is editable
		if (!isEndpoint)
			e->path_points[g_selectedPoint].x = (float)mx;
		e->path_points[g_selectedPoint].y = (float)my;
	}

	return true;
}

// =============================================================================
//  END EDITOR MODE
// =============================================================================


void display()
{
	// Fixed time step
	//static double currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	//static double accumulator = 0.0;

	//double newTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	//double frameTime = newTime - currentTime;
	//currentTime = newTime;

	//if (frameTime > DT * 10.0)
	//	frameTime = DT * 10.0;

	//accumulator += frameTime;

	//while (accumulator >= DT)
	//{
	//	simulate();
	//	accumulator -= DT;
	//	GLOBAL_TIME += DT;
	//}

	static float lastTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f; // Convert to seconds
	float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	//	const float DT = 1.0f / FPS;

	float d = currentTime - lastTime;

	if (DT < d)
	{
		if (!g_editorMode)
		{
			simulate();
			GLOBAL_TIME += DT;
		}
		else
		{
			// Editor mode: left/right arrows scroll the foreground and enemies
			float scrollDelta = 0.0f;
			if (rightKeyPressed) scrollDelta += g_editorFgScrollSpeed * DT;
			if (leftKeyPressed)  scrollDelta -= g_editorFgScrollSpeed * DT;

			if (scrollDelta != 0.0f)
			{
				// Track editor-only scroll so editorSaveToDatabase can strip it
				// from path_points before writing canonical positions.
				g_editorScrollAccum += scrollDelta;

				for (size_t i = 0; i < foreground_chunked.size(); i++)
					foreground_chunked[i].x += scrollDelta;

				for (size_t i = 0; i < foreground_lit_chunked.size(); i++)
					foreground_lit_chunked[i].x += scrollDelta;

				for (size_t i = 0; i < enemy_ships.size(); i++)
				{
					enemy_ships[i]->x += scrollDelta;
					for (size_t j = 0; j < enemy_ships[i]->path_points.size(); j++)
						enemy_ships[i]->path_points[j].x += scrollDelta;
				}
			}

			// Rebuild obstacle texture so foreground collisions stay in sync
			GLuint clearColor[4] = { 0, 0, 0, 0 };
			glClearTexImage(obstacleTex, 0, GL_RGBA, GL_UNSIGNED_BYTE, clearColor);

			if (protagonist.to_be_culled == false)
			{
				addObstacleStamp(protagonist.tex,
					static_cast<int>(protagonist.x), static_cast<int>(protagonist.y),
					protagonist.width, protagonist.height, true,
					1, true);
			}

			for (size_t i = 0; i < foreground_chunked.size(); i++)
			{
				if (foreground_chunked[i].tex != 0 && foreground_chunked[i].isOnscreen())
				{
					addObstacleStamp(foreground_chunked[i].tex,
						static_cast<int>(foreground_chunked[i].x), static_cast<int>(foreground_chunked[i].y),
						foreground_chunked[i].width, foreground_chunked[i].height, true,
						0.5, true);
				}
			}

			for (size_t i = 0; i < enemy_ships.size(); i++)
			{
				if (enemy_ships[i]->tex != 0 && enemy_ships[i]->isOnscreen() && false == enemy_ships[i]->to_be_culled)
				{
					addObstacleStamp(enemy_ships[i]->tex,
						static_cast<int>(enemy_ships[i]->x), static_cast<int>(enemy_ships[i]->y),
						enemy_ships[i]->width, enemy_ships[i]->height, true,
						0.5, true);
				}
			}
		}

		lastTime = currentTime;
	}










	// Render to screen (or scene FBO if glow enabled)
	// ============== GLOW: RENDER SCENE TO OFFSCREEN BUFFER ==============
	if (glowEnabled) {
		glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO);
	}
	else {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	// ============== END GLOW MODIFICATION ==============
	glViewport(0, 0, windowWidth, windowHeight);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(displayProgram);
	setTextureUniform(displayProgram, "density", 0, densityTex[currentDensity]);
	setTextureUniform(displayProgram, "velocity", 1, velocityTex[currentVelocity]);
	setTextureUniform(displayProgram, "obstacles", 2, obstacleTex);
	setTextureUniform(displayProgram, "background", 3, background.tex);
	setTextureUniform(displayProgram, "background_lit", 4, background_lit.tex);
	glUniform1f(glGetUniformLocation(displayProgram, "time"), GLOBAL_TIME);
	glUniform2f(glGetUniformLocation(displayProgram, "texelSize"), 1.0f / windowWidth, 1.0f / windowHeight);

	drawQuad();





	if (protagonist.to_be_culled == false && protagonist.tex != 0)
	{
		drawSprite(protagonist.tex,
			static_cast<int>(protagonist.x), static_cast<int>(protagonist.y),
			protagonist.width, protagonist.height, protagonist.under_fire || (protagonist.last_time_collided > 0 && GLOBAL_TIME <= protagonist.last_time_collided + 0.5), 1.0);
	}

	for (size_t i = 0; i < foreground_chunked.size(); i++)
	{
		if (foreground_chunked[i].tex != 0 && foreground_chunked[i].isOnscreen())
		{
			drawSprite(foreground_chunked[i].tex,
				static_cast<int>(foreground_chunked[i].x), static_cast<int>(foreground_chunked[i].y),
				foreground_chunked[i].width, foreground_chunked[i].height, false, 1.0);
		}
	}

	// Lit foreground overlay — drawn on top of the base foreground using alpha blending.
	// Transparent pixels in foreground_lit.png let the base foreground show through.
	for (size_t i = 0; i < foreground_lit_chunked.size(); i++)
	{
		if (foreground_lit_chunked[i].tex != 0 && foreground_lit_chunked[i].isOnscreen())
		{
			drawSprite(foreground_lit_chunked[i].tex,
				static_cast<int>(foreground_lit_chunked[i].x), static_cast<int>(foreground_lit_chunked[i].y),
				foreground_lit_chunked[i].width, foreground_lit_chunked[i].height, false, alpha);
		}
	}

	for (size_t i = 0; i < enemy_ships.size(); i++)
	{
		if (enemy_ships[i]->isOnscreen() && enemy_ships[i]->to_be_culled == false)
		{
			// Render via chunks (the enemy is visually sliced into a grid
			// just like the foreground). Each chunk owns its own multi-
			// frame data + blackening state; we sync its tilt frame to
			// the parent enemy's state and re-upload if needed.
			if (!enemy_ships[i]->chunks.empty())
			{
				const int ex = static_cast<int>(enemy_ships[i]->x);
				const int ey = static_cast<int>(enemy_ships[i]->y);
				const int parent_state = enemy_ships[i]->state;
				const bool under_fire = enemy_ships[i]->under_fire;

				for (size_t k = 0; k < enemy_ships[i]->chunks.size(); ++k)
				{
					enemy_chunk& c = enemy_ships[i]->chunks[k];
					if (c.tex == 0) continue;

					// Keep the GL texture showing the correct tilt frame.
					// animate_blackening already calls update_tex() on any
					// chunk it touched, but chunks with no hits still need
					// a refresh when the parent's state changes.
					if (c.state != parent_state)
					{
						c.state = parent_state;
						c.update_tex();
					}

					drawSprite(c.tex,
						ex + c.offset_x, ey + c.offset_y,
						c.width, c.height, under_fire, 1.0);
				}
			}
			else if (enemy_ships[i]->tex != 0)
			{
				// Safety fallback: if for some reason chunks were never
				// built (e.g. width/height were 0 at spawn), draw the
				// whole sprite the old way.
				drawSprite(enemy_ships[i]->tex,
					static_cast<int>(enemy_ships[i]->x), static_cast<int>(enemy_ships[i]->y),
					enemy_ships[i]->width, enemy_ships[i]->height, enemy_ships[i]->under_fire, 1.0);
			}
		}
	}

	// Draw live power-ups
	for (size_t i = 0; i < power_ups_alive.size(); i++)
	{
		if (power_ups_alive[i]->tex != 0 && !power_ups_alive[i]->to_be_culled)
		{
			drawSprite(power_ups_alive[i]->tex,
				static_cast<int>(power_ups_alive[i]->x),
				static_cast<int>(power_ups_alive[i]->y),
				power_ups_alive[i]->width, power_ups_alive[i]->height, false, 1.0);
		}
	}




	// 
	// ============== GLOW: APPLY POST-PROCESSING ==============
	if (glowEnabled) {
		applyGlowEffect();
	}
	// ============== END GLOW POST-PROCESSING ==============


		// Draw health bars
	// Protagonist health bar
	if (false == protagonist.to_be_culled && protagonist.tex != 0)
	{
		drawHealthBar(
			static_cast<int>(protagonist.x),
			static_cast<int>(protagonist.y),
			protagonist.width,
			protagonist.health,
			protagonist.max_health);
	}

	// Enemy health bars
	for (size_t i = 0; i < enemy_ships.size(); i++)
	{
		if (enemy_ships[i]->tex != 0 && enemy_ships[i]->isOnscreen() && enemy_ships[i]->to_be_culled == false)
		{
			drawHealthBar(
				static_cast<int>(enemy_ships[i]->x),
				static_cast<int>(enemy_ships[i]->y),
				enemy_ships[i]->width,
				enemy_ships[i]->health,
				enemy_ships[i]->max_health);
		}
	}


	// ============== CHROMATIC ABERRATION: APPLY DAMAGE EFFECT ==============
	applyChromaticAberration();
	// ============== END CHROMATIC ABERRATION ==============


	applyWaveChromaticAberration();


	if (protagonist.to_be_culled == true && game_over_banner.tex != 0)
	{
		drawSprite(game_over_banner.tex,
			0, 0,
			game_over_banner.width, game_over_banner.height, false, 1.0);
	}


	displayFPS();


	for (size_t e = 0; e < enemy_ships.size(); e++)
	{
		if (false == enemy_ships[e]->isOnscreen() && enemy_ships[e]->to_be_culled == true)
			continue;

		glm::vec2 previous_pos = get_spline_point(enemy_ships[e]->path_points, 0.0f);

		for (size_t i = 1; i <= 100; i++)
		{
			float t = i / 100.0f;

			glm::vec2 vd = get_spline_point(enemy_ships[e]->path_points, t);

			lines.push_back(Line(previous_pos, vd, glm::vec4(1, 1, 1, 1)));

			glm::vec2 tangent = get_spline_tangent(enemy_ships[e]->path_points, t);

			float d = get_spline_point(enemy_ships[e]->path_speeds, t);

			tangent.x *= d;
			tangent.y *= d;

			tangent_lines.push_back(Line(vd, (vd + tangent), glm::vec4(0, 1, 0, 1)));

			previous_pos = vd;
		}

		//drawLinesWithWidth(lines, 4.0f);
		//drawLinesWithWidth(tangent_lines, 4.0f);

	}







	//std::vector<Point> pv;

	//if (enemy_ships.size() > 0)
	//	for (size_t i = 0; i < enemy_ships[0]->path_points.size(); i++)
	//	{
	//		Point p(
	//			enemy_ships[0]->path_points[i].x,
	//			enemy_ships[0]->path_points[i].y,
	//			glm::vec4(1, 0, 0, 1));

	//		//cout << p.position.x << " " << p.position.y << endl;

	//		pv.push_back(p);
	//	}

	//drawPointsWithSize(pv, 20.0f);









	//	lines.clear();
	//
	//	for (size_t i = 0; i < enemy_ships.size(); i++)
	//	{
	//		for (size_t j = 0; j < enemy_ships[i]->path_points.size() - 1; j++)
	//		{
	//			glm::vec2 p0 = enemy_ships[i]->path_points[j];
	//
	//			//p0.y = 1.0f - p0.y;
	//			//p0.x *= SIM_WIDTH;
	//
	//			p0.x *= SIM_WIDTH;
	//			p0.y *= SIM_HEIGHT;
	//
	//
	//
	//			//p0.x += enemy_ships[i]->x;
	//			//p0.y += enemy_ships[i]->y;
	//			//p0.y = 1080 - 1 - p0.y;
	//
	//			//p0.y = 1080 - 1 - p0.y;
	//
	//			glm::vec2 p1 = enemy_ships[i]->path_points[j + 1];
	//
	//			//p1.y = 1.0f - p1.y;
	//			//p1.x *= SIM_WIDTH;
	//
	//		//	p1.x = -enemy_ships[i]->width;
	//			p1.x = -enemy_ships[i]->width;
	//			p1.y *= SIM_HEIGHT;
	//			//p1.x += enemy_ships[i]->x;
	//			//p1.y += enemy_ships[i]->y;
	//			//p1.y = 1080 - 1 - p1.y;
	//
	//			//p1.y = 1080 - 1 - enemy_ships[i]->y;
	//
	//			//p1.y = 1080 - 1 - p1.y;
	//
	//			lines.push_back(Line(p0, p1, glm::vec4(1, 1, 1, 1)));
	//		}
	//	}
	//
	//drawLinesWithWidth(lines, 1.0f);





			//// Add continuous sources based on mouse input
			//if (leftMouseDown && !shiftDown) {
			//	float x = (float)mouseX / windowWidth;
			//	float y = 1.0f - (float)mouseY / windowHeight;

			//	if (red_mode)
			//		addSource(densityTex, densityFBO, currentDensity, x, y, 1, 0, 0, 0.0008f);
			//	else
			//		addSource(densityTex, densityFBO, currentDensity, x, y, 0, 1, 0, 0.0008f);
			//}

			//if (rightMouseDown) {
			//	float x = (float)mouseX / windowWidth;
			//	float y = 1.0f - (float)mouseY / windowHeight;
			//	float dx = (float)(mouseX - lastMouseX) * 2.0f;
			//	float dy = (float)(lastMouseY - mouseY) * 2.0f;

			//	addSource(velocityTex, velocityFBO, currentVelocity, x, y, dx, dy, 0.0f, 0.00008f);
			//}


	lastMouseX = mouseX;
	lastMouseY = mouseY;

	// Detect fluid-obstacle collisions
	static int collision_lastCallTime = 0;
	int curr_time_int = glutGet(GLUT_ELAPSED_TIME);

	if (curr_time_int - collision_lastCallTime >= COLLISION_INTERVAL_MS)
	{
		detectEdgeCollisions();
		collision_lastCallTime = curr_time_int;
	}

	for (auto it = ally_bullets.begin(); it != ally_bullets.end();)
	{
		if ((*it)->to_be_culled)
		{
			//cout << "culling ally bullet" << endl;
			it = ally_bullets.erase(it);
		}
		else
			it++;
	}

	for (auto it = enemy_bullets.begin(); it != enemy_bullets.end();)
	{
		if ((*it)->to_be_culled)
		{
			cout << "culling enemy bullet" << endl;
			it = enemy_bullets.erase(it);
		}
		else
			it++;
	}

	for (auto it = power_ups_alive.begin(); it != power_ups_alive.end();)
	{
		if ((*it)->to_be_culled)
		{
			cout << "culling power up" << endl;
			it = power_ups_alive.erase(it);
		}
		else
			++it;
	}

	//for (auto it = enemy_ships.begin(); it != enemy_ships.end();)
	//{
	//	if (!g_editorMode && (*it)->to_be_culled)
	//	{
	//		cout << "culling enemy enemy_ship" << endl;
	//		it = enemy_ships.erase(it);
	//	}
	//	else
	//		it++;
	//}


	renderEditorOverlay();

	glutSwapBuffers();
	glutPostRedisplay();
}

void reshape(int w, int h) {
	windowWidth = w;
	windowHeight = h;

	// ============== GLOW: RESIZE BUFFERS ==============
	// Recreate glow buffers when window is resized
	resizeGlowResources();
	// ============== END GLOW RESIZE ==============
}

void keyboard(unsigned char key, int x, int y)
{
	if (editorHandleKey(key, x, y)) return;

	switch (key)
	{
	case ' ': // Space bar
		spacePressed = true;
		break;

	case 'x':
	{
		red_mode = !red_mode;
		break;
	}



	//case 27:  // ESC
	//case 'q':
	//case 'Q':
	//	exit(0);
	//	break;
	case 'r':
	case 'R':
		// Reset simulation
		for (int i = 0; i < 2; i++) {
			glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[i]);
			glClear(GL_COLOR_BUFFER_BIT);
			glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[i]);
			glClear(GL_COLOR_BUFFER_BIT);
			glBindFramebuffer(GL_FRAMEBUFFER, densityFBO[i]);
			glClear(GL_COLOR_BUFFER_BIT);
		}
		std::cout << "Simulation reset" << std::endl;
		break;
	case 'o':
	case 'O':
		// Clear obstacles
		glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
		glClear(GL_COLOR_BUFFER_BIT);
		std::cout << "Obstacles cleared" << std::endl;
		break;

		// ============== GLOW CONTROLS ==============
	case 'g':
	case 'G':
		glowEnabled = !glowEnabled;
		std::cout << "Glow: " << (glowEnabled ? "ON" : "OFF") << std::endl;
		break;

	case '+':
	case '=':
		glowIntensity += 0.1f;
		std::cout << "Glow intensity: " << glowIntensity << std::endl;
		break;

	case '-':
	case '_':
		glowIntensity = std::max(0.0f, glowIntensity - 0.1f);
		std::cout << "Glow intensity: " << glowIntensity << std::endl;
		break;

	case '[':
		glowThreshold = std::max(0.0f, glowThreshold - 0.05f);
		std::cout << "Glow threshold: " << glowThreshold << std::endl;
		break;

	case ']':
		glowThreshold = std::min(1.0f, glowThreshold + 0.05f);
		std::cout << "Glow threshold: " << glowThreshold << std::endl;
		break;

	case ',':
		glowBlurPasses = std::max(1, glowBlurPasses - 1);
		std::cout << "Glow blur passes: " << glowBlurPasses << std::endl;
		break;

	case '.':
		glowBlurPasses = std::min(10, glowBlurPasses + 1);
		std::cout << "Glow blur passes: " << glowBlurPasses << std::endl;
		break;
		// ============== END GLOW CONTROLS ==============

	// ============== CHROMATIC ABERRATION CONTROLS ==============
	case 'c':
	case 'C':
		chromaticAberrationEnabled = !chromaticAberrationEnabled;
		std::cout << "Chromatic Aberration: " << (chromaticAberrationEnabled ? "ON" : "OFF") << std::endl;
		break;

	case '1':
		aberrationIntensity = std::max(0.0f, aberrationIntensity - 0.005f);
		std::cout << "Aberration intensity: " << aberrationIntensity << std::endl;
		break;

	case '2':
		aberrationIntensity = std::min(0.1f, aberrationIntensity + 0.005f);
		std::cout << "Aberration intensity: " << aberrationIntensity << std::endl;
		break;

	case '3':
		vignetteStrength = std::max(0.0f, vignetteStrength - 0.1f);
		std::cout << "Vignette strength: " << vignetteStrength << std::endl;
		break;

	case '4':
		vignetteStrength = std::min(1.0f, vignetteStrength + 0.1f);
		std::cout << "Vignette strength: " << vignetteStrength << std::endl;
		break;

	case '5':
		// Test trigger - manually activate the effect
		lastDamageTime = GLOBAL_TIME;
		std::cout << "Chromatic aberration triggered manually!" << std::endl;
		break;
		// ============== END CHROMATIC ABERRATION CONTROLS ==============
	}
}

void specialKeys(int key, int x, int y) {

	bool shiftHeld = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;

	// In editor mode, Shift+Arrow nudges the selected enemy AND its path points.
	// We handle this here as a discrete step and return early so the foreground
	// scroll logic (which runs continuously in the display loop) is not triggered.
	if (g_editorMode && shiftHeld)
	{
		enemy_ship* e = editorSelected();
		if (e)
		{
			const float NUDGE = 5.0f; // pixels per key-press; increase for coarser steps
			float dx = 0.0f, dy = 0.0f;
			switch (key)
			{
			case GLUT_KEY_UP:    dy = -NUDGE; break;
			case GLUT_KEY_DOWN:  dy = NUDGE; break;
			case GLUT_KEY_LEFT:  dx = -NUDGE; break;
			case GLUT_KEY_RIGHT: dx = NUDGE; break;
			}
			if (dx != 0.0f || dy != 0.0f)
			{
				editorPushUndo();
				// Move the sprite position
				e->x += dx;
				e->y += dy;
				// Move every path control point so the spline travels with the enemy
				for (auto& pt : e->path_points)
				{
					pt.x += dx;
					pt.y += dy;
				}
				std::cout << "[Editor] Nudged enemy " << g_selectedEnemy
					<< " by (" << dx << ", " << dy << ")  "
					<< "pos=(" << e->x << ", " << e->y << ")\n";
			}
		}
		return; // do NOT set key flags — foreground scrolls only on plain Arrow
	}

	switch (key) {
	case GLUT_KEY_UP:
		upKeyPressed = true;
		break;
	case GLUT_KEY_DOWN:
		downKeyPressed = true;
		break;
	case GLUT_KEY_LEFT:
		leftKeyPressed = true;
		break;
	case GLUT_KEY_RIGHT:
		rightKeyPressed = true;
		break;
	}

	// In editor mode, left/right scroll the foreground (handled in display loop)
	if (g_editorMode) return;

	float local_vel_x = 0;
	float local_vel_y = 0;

	// Combine key states to allow diagonal movement
	if (upKeyPressed) {
		local_vel_y = -1;
	}
	if (downKeyPressed) {
		local_vel_y = 1;
	}
	if (leftKeyPressed) {
		local_vel_x = -1;
	}
	if (rightKeyPressed) {
		local_vel_x = 1;
	}

	float vel_length = sqrt(local_vel_x * local_vel_x + local_vel_y * local_vel_y);

	if (vel_length > 0)
	{
		local_vel_x /= vel_length * 2;
		local_vel_y /= vel_length * 2;
	}

	protagonist.set_target_velocity(local_vel_x * windowWidth, local_vel_y * windowHeight);
}

void specialKeysUp(int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_UP:
		upKeyPressed = false;
		break;
	case GLUT_KEY_DOWN:
		downKeyPressed = false;
		break;
	case GLUT_KEY_LEFT:
		leftKeyPressed = false;
		break;
	case GLUT_KEY_RIGHT:
		rightKeyPressed = false;
		break;


	}

	// In editor mode, left/right scroll the foreground (handled in display loop)
	if (g_editorMode) return;

	float local_vel_x = 0;
	float local_vel_y = 0;

	// Combine key states to allow diagonal movement
	if (upKeyPressed) {
		local_vel_y = -1;
	}
	if (downKeyPressed) {
		local_vel_y = 1;
	}
	if (leftKeyPressed) {
		local_vel_x = -1;
	}
	if (rightKeyPressed) {
		local_vel_x = 1;
	}

	float vel_length = sqrt(local_vel_x * local_vel_x + local_vel_y * local_vel_y);

	if (vel_length > 0)
	{
		local_vel_x /= vel_length * 2;
		local_vel_y /= vel_length * 2;
	}

	protagonist.set_target_velocity(local_vel_x * windowWidth, local_vel_y * windowHeight);
}

void mouse(int button, int state, int x, int y) {
	if (editorHandleMouse(button, state, x, y)) return;

	mouseX = x;
	mouseY = y;

	shiftDown = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;

	if (button == GLUT_LEFT_BUTTON) {
		leftMouseDown = (state == GLUT_DOWN);
	}
	else if (button == GLUT_RIGHT_BUTTON) {
		rightMouseDown = (state == GLUT_DOWN);
	}
	else if (button == GLUT_MIDDLE_BUTTON) {
		middleMouseDown = (state == GLUT_DOWN);
	}
}

void motion(int x, int y) {
	if (editorHandleMotion(x, y)) return;

	mouseX = x;
	mouseY = y;
}

void passiveMotion(int x, int y) {
	lastMouseX = mouseX;
	lastMouseY = mouseY;
	mouseX = x;
	mouseY = y;
}



void specialKeyboard(int key, int x, int y)
{
	bool shiftHeld = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;

	// In editor mode, Shift+Arrow nudges the selected enemy AND its path points.
	// We handle this here as a discrete step and return early so the foreground
	// scroll logic (which runs continuously in the display loop) is not triggered.
	if (g_editorMode && shiftHeld)
	{
		enemy_ship* e = editorSelected();
		if (e)
		{
			const float NUDGE = 5.0f; // pixels per key-press; increase for coarser steps
			float dx = 0.0f, dy = 0.0f;
			switch (key)
			{
			case GLUT_KEY_UP:    dy = -NUDGE; break;
			case GLUT_KEY_DOWN:  dy = NUDGE; break;
			case GLUT_KEY_LEFT:  dx = -NUDGE; break;
			case GLUT_KEY_RIGHT: dx = NUDGE; break;
			}
			if (dx != 0.0f || dy != 0.0f)
			{
				editorPushUndo();
				// Move the sprite position
				e->x += dx;
				e->y += dy;
				// Move every path control point so the spline travels with the enemy
				for (auto& pt : e->path_points)
				{
					pt.x += dx;
					pt.y += dy;
				}
				std::cout << "[Editor] Nudged enemy " << g_selectedEnemy
					<< " by (" << dx << ", " << dy << ")  "
					<< "pos=(" << e->x << ", " << e->y << ")\n";
			}
		}
		return; // do NOT set key flags — foreground scrolls only on plain Arrow
	}

	switch (key) {
	case GLUT_KEY_UP:
		upKeyPressed = true;
		break;
	case GLUT_KEY_DOWN:
		downKeyPressed = true;
		break;
	case GLUT_KEY_LEFT:
		leftKeyPressed = true;
		break;
	case GLUT_KEY_RIGHT:
		rightKeyPressed = true;
		break;
	}

	// In editor mode, left/right scroll the foreground (handled in display loop)
	if (g_editorMode) return;

	float local_vel_x = 0;
	float local_vel_y = 0;

	// Combine key states to allow diagonal movement
	if (upKeyPressed) {
		local_vel_y = -1;
	}
	if (downKeyPressed) {
		local_vel_y = 1;
	}
	if (leftKeyPressed) {
		local_vel_x = -1;
	}
	if (rightKeyPressed) {
		local_vel_x = 1;
	}

	float vel_length = sqrt(local_vel_x * local_vel_x + local_vel_y * local_vel_y);

	if (vel_length > 0)
	{
		local_vel_x /= vel_length * 2;
		local_vel_y /= vel_length * 2;
	}

	protagonist.set_target_velocity(local_vel_x * windowWidth, local_vel_y * windowHeight);
}

// Modified specialKeyboardUp function to reset key states
void specialKeyboardUp(int key, int x, int y)
{
	switch (key) {
	case GLUT_KEY_UP:
		upKeyPressed = false;
		break;
	case GLUT_KEY_DOWN:
		downKeyPressed = false;
		break;
	case GLUT_KEY_LEFT:
		leftKeyPressed = false;
		break;
	case GLUT_KEY_RIGHT:
		rightKeyPressed = false;
		break;


	}

	// In editor mode, left/right scroll the foreground (handled in display loop)
	if (g_editorMode) return;

	float local_vel_x = 0;
	float local_vel_y = 0;

	// Combine key states to allow diagonal movement
	if (upKeyPressed) {
		local_vel_y = -1;
	}
	if (downKeyPressed) {
		local_vel_y = 1;
	}
	if (leftKeyPressed) {
		local_vel_x = -1;
	}
	if (rightKeyPressed) {
		local_vel_x = 1;
	}

	float vel_length = sqrt(local_vel_x * local_vel_x + local_vel_y * local_vel_y);

	if (vel_length > 0)
	{
		local_vel_x /= vel_length * 2;
		local_vel_y /= vel_length * 2;
	}

	protagonist.set_target_velocity(local_vel_x * windowWidth, local_vel_y * windowHeight);
}

void keyboardup(unsigned char key, int x, int y) {
	switch (key) {
	case ' ': // Space bar

		spacePressed = false;


		break;
	}
}










int main(int argc, char** argv)
{
	// Initialize GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitContextVersion(4, 0);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutCreateWindow("2D Navier-Stokes Fluid Simulation");

	// Initialize GLEW
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cerr << "GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
		return -1;
	}

	// Clear any errors from GLEW init
	while (glGetError() != GL_NO_ERROR);

	std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;


	textRenderer = new TextRenderer("media/font.png", windowWidth, windowHeight);

	// Initialize
	initShaders();
	initTextures();
	initQuad();

	// ============== GLOW: INITIALIZE RESOURCES ==============
	initGlowResources();  // Initialize glow effect resources
	// ============== END GLOW INIT ==============

	initSplatBatchResources();

	initLineRenderer();
	initPointRenderer();



	initCollisionResources();



	load_media("level1");



	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyboardup);
	glutSpecialFunc(specialKeys);
	glutSpecialUpFunc(specialKeysUp);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(passiveMotion);
	//glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF);
	glutFullScreen();

	// Main loop


	glutMainLoop();

	return 0;
}