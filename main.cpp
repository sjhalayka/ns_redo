#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
using namespace std;


std::mt19937 generator_real(static_cast<unsigned>(0));
std::uniform_real_distribution<double> dis_real(0, 1);




bool red_mode = true;

float GLOBAL_TIME = 0;
const float FPS = 30;
float DT = 1.0f / FPS;
const int COLLISION_INTERVAL_MS = 100; // 100ms = 10 times per second

// Simulation parameters
const int SIM_WIDTH = 1920;
const int SIM_HEIGHT = 1080;
const int JACOBI_ITERATIONS = 20;
const float DENSITY_DISSIPATION = 0.95f;
const float VELOCITY_DISSIPATION = 0.95f;
const float VORTICITY_SCALE = 0.1f;

float TURBULENCE_AMPLITUDE = 2.0f;      // Controls noise strength
float TURBULENCE_FREQUENCY = 10.0f;      // Controls noise frequency (scale)
float TURBULENCE_SCALE = 0.05f;          // Overall turbulence strength




bool spacePressed = false;

const float MIN_BULLET_INTERVAL = 0.5f;

// Add a variable to track the time of the last fired bullet
std::chrono::high_resolution_clock::time_point lastBulletTime = std::chrono::high_resolution_clock::now();

bool x3_fire = false;
bool x5_fire = true;


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


void RandomUnitVector(double& x_out, double& y_out)
{
	const static double pi = 4.0 * atan(1.0);

	const double a = dis_real(generator_real) * 2.0f * pi;

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

	bool isOnscreen(void)
	{
		return
			(x + width > 0) &&
			(x < windowWidth) &&
			(y + height > 0) &&
			(y < windowHeight);
	}

	void integrate(float dt)
	{
		old_x = x;
		old_y = y;

		x = x + vel_x * dt;
		y = y + vel_y * dt;
	}

	void animate_blackening(const vector<glm::vec2>& locations)
	{
		float glut_curr_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

//		blackening_age_map.clear();

		for (size_t i = 0; i < locations.size(); i++)
			blackening_age_map[locations[i]] = glut_curr_time;

		for (size_t i = 0; i < to_present_data_pointers.size(); i++)
		{
			if (to_present_data_pointers[i] == 0)
				continue;

			for (map<glm::vec2, float>::const_iterator ci = blackening_age_map.begin(); ci != blackening_age_map.end(); ci++)
			{
				glm::vec2 point(ci->first.x, ci->first.y);

				const float BRUSH_RADIUS = 15.0f;        // Radius of the soft brush in sprite pixels
				const float BRUSH_RADIUS_SQUARED = BRUSH_RADIUS * BRUSH_RADIUS;

				int minX = std::max(0, (int)(point.x - BRUSH_RADIUS - 1));
				int maxX = std::min(width - 1, (int)(point.x + BRUSH_RADIUS + 1));
				int minY = std::max(0, (int)(point.y - BRUSH_RADIUS - 1));
				int maxY = std::min(height - 1, (int)(point.y + BRUSH_RADIUS + 1));

				bool transparent = false;

				if (dis_real(generator_real) > 0.99999)
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

							const float animation_length = 5.0;

							if (duration >= animation_length)
							{
								to_present_data_pointers[i][index + 0] = 0;
								to_present_data_pointers[i][index + 1] = 0;
								to_present_data_pointers[i][index + 2] = 0;	
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
		to_present_data_pointers.push_back(&to_present_data[0]);
	}

	sprite(void)
	{
		to_present_data_pointers.push_back(&to_present_data[0]);
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

	std::vector<unsigned char> to_present_up_data;
	std::vector<unsigned char> to_present_down_data;
	std::vector<unsigned char> to_present_rest_data;

	int state = REST_STATE;

	void rebuild_pointers()
	{
		to_present_data_pointers.clear();
		to_present_data_pointers.push_back(&to_present_up_data[0]);
		to_present_data_pointers.push_back(&to_present_down_data[0]);
		to_present_data_pointers.push_back(&to_present_rest_data[0]);
	}

	tri_sprite(void)
	{

	}

	void manually_update_data(
		const std::vector<unsigned char> &src_to_present_up_data,
		const std::vector<unsigned char> &src_to_present_down_data,
		const std::vector<unsigned char> &src_to_present_rest_data)
	{
		if(src_to_present_up_data.size() > 0)
			to_present_up_data = src_to_present_up_data;

		if (src_to_present_down_data.size() > 0)
			to_present_down_data = src_to_present_down_data;
		
		if (src_to_present_rest_data.size() > 0)
			to_present_rest_data = src_to_present_rest_data;

		rebuild_pointers();
		update_tex();
	}



	//----------------------------------------------------------------------
	//  Update OpenGL texture based on state
	//----------------------------------------------------------------------
	void update_tex()
	{
		glBindTexture(GL_TEXTURE_2D, tex);

		if (state == UP_STATE)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
				GL_RGBA, GL_UNSIGNED_BYTE,
				to_present_up_data.data());

		else if (state == DOWN_STATE)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
				GL_RGBA, GL_UNSIGNED_BYTE,
				to_present_down_data.data());

		else if (state == REST_STATE)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
				GL_RGBA, GL_UNSIGNED_BYTE,
				to_present_rest_data.data());
	}
};



class ship : public tri_sprite
{
public:

	float health;
};




class friendly_ship : public ship
{
public:

	void set_velocity(const float src_x, const float src_y)
	{
		vel_x = src_x;
		vel_y = src_y;

		if (vel_y < 0)
		{
			state = UP_STATE;
		}
		else if (vel_y > 0)
		{
			state = DOWN_STATE;
		}
		else
		{
			state = REST_STATE;
		}

		update_tex();
	}

};


class enemy_ship : public ship
{
public:

	void set_velocity(const float src_x, const float src_y)
	{
	}

	float health;
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

	virtual void integrate(float dt)
	{


	}
};

class sine_bullet : public bullet
{
public:
	float sinusoidal_frequency;
	float sinusoidal_amplitude;
	bool sinusoidal_shift;

	void integrate(float dt)
	{
		const float inv_aspect = SIM_HEIGHT / float(SIM_WIDTH);

		old_x = x;
		old_y = y;

		//std::chrono::high_resolution_clock::time_point global_time_end = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<float, std::milli> elapsed;
		//elapsed = global_time_end - app_start_time;

		// Store the original direction vector
		float dirX = vel_x * inv_aspect * dt;
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
		float timeSinceCreation = GLOBAL_TIME - birth_time;
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
		x += perpX * sinValue * amplitude * dt;// *(120.0f / FPS);
		y += perpY * sinValue * amplitude * dt;// *(120.0f / FPS);

		//float path_randomization = 10;// dis_real(generator_real) * 0.01f;
		//float rand_x = 0, rand_y = 0;
		//RandomUnitVector(rand_x, rand_y);
		//x += rand_x * path_randomization;
		//y += rand_y * path_randomization;
	}

};

inline float pixelToNormX(float px) { return px / (float)SIM_WIDTH; }
inline float pixelToNormY(float py) { return 1.0f - py / (float)SIM_HEIGHT; }  // Flip Y

// Convert pixel velocity to normalized velocity  
inline float velPixelToNormX(float vx) { return vx / (float)SIM_WIDTH; }
inline float velPixelToNormY(float vy) { return -vy / (float)SIM_HEIGHT; }

vector<unique_ptr<bullet>> ally_bullets;

vector<unique_ptr<enemy_ship>> enemy_ships;



// These global objects do not need to generate new tex(es)
friendly_ship protagonist;
background_tile background;
bullet bullet_template;
enemy_ship enemy0_template;
enemy_ship enemy1_template;
boss_ship boss_template;

const int foreground_chunk_size_width = 360;
const int foreground_chunk_size_height = 108;


vector<foreground_tile> foreground_chunked;



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




// Quad VAO
GLuint quadVAO, quadVBO;







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
	if (spr.state == UP_STATE && !spr.to_present_up_data.empty())
		return spr.to_present_up_data.data();
	else if (spr.state == DOWN_STATE && !spr.to_present_down_data.empty())
		return spr.to_present_down_data.data();
	else if (!spr.to_present_rest_data.empty())
		return spr.to_present_rest_data.data();
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



TextRenderer* textRenderer = nullptr;


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
        fragColor = vec4(0.0);
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

    if(length(redFluidColor.r) > 0.5)
        color4 = vec4(0.0, 0.0, 0.0, 0.0);
    else
        color4 = vec4(1.0, 1.0, 1.0, 0.0);

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

uniform sampler2D spriteTexture;
uniform int under_fire;
uniform float time;

void main() {
    vec4 color = texture(spriteTexture, texCoord);
    // Discard fully transparent pixels
    //if (color.a < 0.01) discard;


	// Do alternating colour / white blinking when under fire
	if(under_fire == 1)
	{
		const float timeslice = 0.25;
		float m = mod(time, timeslice);
		
		if(m < timeslice/2.0)
		color.rgb = vec3(1.0, 1.0, 1.0);
	}


    fragColor = color;
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

	// Draw
	glUseProgram(lineProgram);
	glUniform2f(glGetUniformLocation(lineProgram, "resolution"),
		(float)windowWidth, (float)windowHeight);

	glBindVertexArray(lineVAO);
	glDrawArrays(GL_LINES, 0, (GLsizei)vertices.size());
	glBindVertexArray(0);
}

// Optional: Draw with custom line width
void drawLinesWithWidth(const std::vector<Line>& linesToDraw, float width) {
	glLineWidth(width);
	drawLines(linesToDraw);
	glLineWidth(1.0f);  // Reset to default
}


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
bool isPixelInsideSpriteAndTransparent(
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
	std::vector<unsigned char> texData(sprW * sprH * 4);

	glBindTexture(GL_TEXTURE_2D, sprTex);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, texData.data());

	// 4. Index into pixel buffer
	int idx = (texY * sprW + texX) * 4;
	unsigned char a = texData[idx + 3];  // ALPHA

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

	// Step 3: Collect all non-zero collision points
	for (int y = 0; y < SIM_HEIGHT; ++y)
	{
		for (int x = 0; x < SIM_WIDTH; ++x)
		{
			size_t idx = y * SIM_WIDTH + x;
			glm::vec2 dens = pixelData[idx];

			// If either red or green density is present -> collision
			if (dens.r > 0.1 || dens.g > 0.1)
			{
				if (dens.r > 1)
					dens.r = 1;

				if (dens.g > 1)
					dens.g = 1;

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

				if (isPixelInsideSpriteAndTransparent(
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
					if (inside && collisionPoints[i].w == 1)
					{
						protagonist.under_fire = true;
						protagonist_blackening_points.push_back(glm::vec2(hit.x, hit.y));
					}
				}
			}

			protagonist.animate_blackening(protagonist_blackening_points);
		}

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

			foreground_chunked[h].animate_blackening(blackening_points);
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

				if (isPixelInsideSpriteAndTransparent(
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
					if (inside && collisionPoints[i].z == 1)
					{
						enemy_ships[h]->under_fire = true;
						blackening_points.push_back(glm::vec2(hit.x, hit.y));
					}
				}
			}

			enemy_ships[h]->animate_blackening(blackening_points);
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
GLuint loadTextureFromFile(const char* filename, int* outWidth, int* outHeight, vector<unsigned char>& out_data) {
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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	for (size_t i = 0; i < width * height * channels; i++)
		out_data.push_back(data[i]);

	stbi_image_free(data);

	if (outWidth) *outWidth = width;
	if (outHeight) *outHeight = height;

	std::cout << "Loaded texture: " << filename << " (" << width << "x" << height << ")" << std::endl;
	return tex;
}



GLuint loadTextureFromFile_Triplet(
	const char* up_filename,
	const char* down_filename,
	const char* rest_filename,
	int* outWidth, int* outHeight,
	vector<unsigned char>& out_up_data,
	vector<unsigned char>& out_down_data,
	vector<unsigned char>& out_rest_data,
	tri_sprite& t)
{
	int width, height, channels;

	out_up_data.clear();
	out_down_data.clear();
	out_rest_data.clear();

	// stb_image loads with (0,0) at top-left, which matches our coordinate system
	stbi_set_flip_vertically_on_load(0);  // Don't flip - we handle it in the shader


	unsigned char* up_data = stbi_load(up_filename, &width, &height, &channels, 4);  // Force RGBA
	if (!up_data) {
		std::cerr << "Failed to load texture: " << up_filename << std::endl;
		std::cerr << "stb_image error: " << stbi_failure_reason() << std::endl;
		if (outWidth) *outWidth = 0;
		if (outHeight) *outHeight = 0;
		return 0;
	}

	unsigned char* down_data = stbi_load(down_filename, &width, &height, &channels, 4);  // Force RGBA
	if (!down_data) {
		std::cerr << "Failed to load texture: " << down_filename << std::endl;
		std::cerr << "stb_image error: " << stbi_failure_reason() << std::endl;
		if (outWidth) *outWidth = 0;
		if (outHeight) *outHeight = 0;
		return 0;
	}

	unsigned char* rest_data = stbi_load(rest_filename, &width, &height, &channels, 4);  // Force RGBA
	if (!rest_data) {
		std::cerr << "Failed to load texture: " << rest_filename << std::endl;
		std::cerr << "stb_image error: " << stbi_failure_reason() << std::endl;
		if (outWidth) *outWidth = 0;
		if (outHeight) *outHeight = 0;
		return 0;
	}

	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rest_data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);



	for (size_t i = 0; i < width * height * channels; i++)
		out_up_data.push_back(up_data[i]);

	stbi_image_free(up_data);

	for (size_t i = 0; i < width * height * channels; i++)
		out_down_data.push_back(down_data[i]);

	stbi_image_free(down_data);

	for (size_t i = 0; i < width * height * channels; i++)
		out_rest_data.push_back(rest_data[i]);

	stbi_image_free(rest_data);

	if (outWidth) *outWidth = width;
	if (outHeight) *outHeight = height;

	t.rebuild_pointers();

	return tex;
}




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
bool chunkForegroundTexture(const char* sourceFilename)
{
	foreground_chunked.clear();

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

				foreground_chunked.push_back(tile);
			}
		}
	}

	stbi_image_free(srcData);

	std::cout << "Created " << foreground_chunked.size() << " foreground tiles" << std::endl;
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
void drawSprite(GLuint texture, int pixelX, int pixelY, int pixelWidth, int pixelHeight, bool under_fire) {
	if (texture == 0) return;

	// Enable blending for transparency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glUniform1i(glGetUniformLocation(spriteProgram, "spriteTexture"), 0);

	glUniform1i(glGetUniformLocation(spriteProgram, "under_fire"), under_fire);
	glUniform1f(glGetUniformLocation(spriteProgram, "time"), GLOBAL_TIME);


	drawQuad();

	glDisable(GL_BLEND);
}


void fireBullet(void)
{
	std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> timeSinceLastBullet = currentTime - lastBulletTime;

	if (timeSinceLastBullet.count() < MIN_BULLET_INTERVAL)
		return;

	lastBulletTime = currentTime;

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

	if (x3_fire) { angle_start = 0.1f; angle_end = -0.1f; num_streams = 3; }
	if (x5_fire) { angle_start = 0.2f; angle_end = -0.2f; num_streams = 5; }

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
		newBullet.sinusoidal_amplitude = 300 / DT;  // amplitude in PIXELS
		newBullet.sinusoidal_frequency = 10;
		newBullet.birth_time = GLOBAL_TIME;
		newBullet.death_time = -1;

		ally_bullets.push_back(make_unique<sine_bullet>(newBullet));

		newBullet.sinusoidal_shift = true;
		ally_bullets.push_back(make_unique<sine_bullet>(newBullet));
	}
}


void simulate()
{

	if (spacePressed)
		fireBullet();


	protagonist.integrate(DT);


	//for (size_t i = 0; i < foreground_chunked.size(); i++)
	//{
	//	if (false == foreground_chunked[i].isOnscreen())
	//		continue;

	//	bool found_collision = detectTriSpriteToSpriteOverlap(protagonist, foreground_chunked[i], 1);

	//	if (true == found_collision)
	//	{
	//		friendly_ship old_protagonist = protagonist;
	//		old_protagonist.x = old_protagonist.old_x;
	//		old_protagonist.y = old_protagonist.old_y;

	//		//float x_move = (protagonist.x - protagonist.old_x);
	//		//float y_move = (protagonist.y - protagonist.old_y);

	//		friendly_ship old_protagonist_up = old_protagonist;
	//		old_protagonist_up.y--;// -= y_move;

	//		friendly_ship old_protagonist_down = old_protagonist;
	//		old_protagonist_down.y++;// += y_move;

	//		friendly_ship old_protagonist_left = old_protagonist;
	//		old_protagonist_left.x--;// -= x_move;

	//		friendly_ship old_protagonist_right = old_protagonist;
	//		old_protagonist_right.x++;// += x_move;

	//		bool found_collision_up = detectTriSpriteToSpriteOverlap(old_protagonist_up, foreground_chunked[i], 1);
	//		bool found_collision_down = detectTriSpriteToSpriteOverlap(old_protagonist_down, foreground_chunked[i], 1);
	//		bool found_collision_left = detectTriSpriteToSpriteOverlap(old_protagonist_left, foreground_chunked[i], 1);
	//		bool found_collision_right = detectTriSpriteToSpriteOverlap(old_protagonist_right, foreground_chunked[i], 1);

	//		if (/*found_collision_up ||*/ old_protagonist_up.vel_y < 0)
	//		{
	//			//protagonist.vel_y = 0;// -old_protagonist.vel_y * 0.01;

	//			protagonist.y = protagonist.old_y;
	//		}

	//		if (/*found_collision_down || */old_protagonist_down.vel_y > 0)
	//		{
	//			//protagonist.vel_y = 0;// -old_protagonist.vel_y * 0.01;
	//			protagonist.y = protagonist.old_y;
	//		}

	//		if (/*found_collision_left || */old_protagonist_left.vel_x < 0)
	//		{
	//			//protagonist.vel_x = 0;// -old_protagonist.vel_x * 0.01;
	//			protagonist.x = protagonist.old_x;
	//		}

	//		if (/*found_collision_right || */old_protagonist_right.vel_x > 0)
	//		{
	//			//protagonist.vel_x = 0;// -old_protagonist.vel_x * 0.01;
	//			protagonist.x = protagonist.old_x;
	//		}

	//		//protagonist.x = protagonist.old_x;
	//		//protagonist.y = protagonist.old_y;
	//		

	//		break;
	//	}
	//}





	bool resolved = false;

	for (size_t i = 0; i < foreground_chunked.size() && !resolved; i++)
	{
		if (false == foreground_chunked[i].isOnscreen())
			continue;

		if (detectTriSpriteToSpriteOverlap(protagonist, foreground_chunked[i], 1))
		{
			// to do: do damage to protagonist

			// Test X resolution
			float tempX = protagonist.x;
			protagonist.x = protagonist.old_x;

			bool xResolves = true;
			// Check if moving only X back resolves ALL collisions
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
				continue;
			}

			// Neither alone works -> full revert (corner)
			protagonist.x = protagonist.old_x;
			protagonist.y = protagonist.old_y;
			protagonist.vel_x = 0;
			protagonist.vel_y = 0;
			resolved = true;

			// to do: kill protagonist if still colliding
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
				if (false == enemy_ships[i]->isOnscreen())
					continue;

				found_collision = detectTriSpriteToSpriteOverlap(*enemy_ships[i], *(*it), 1);

				if (true == found_collision)
					break;
			}
		}


		if (false == (*it)->isOnscreen() || found_collision)
		{
			cout << "culling ally bullet" << endl;
			(*it)->to_be_culled = true;
		}
	}

	for (size_t i = 0; i < ally_bullets.size(); i++)
	{
		auto& bullet = ally_bullets[i];

		int pathSamples = 10;

		float prevX = bullet->old_x;
		float prevY = bullet->old_y;

		for (int step = 0; step <= pathSamples; step++)
		{
			float t = static_cast<float>(step) / pathSamples;

			float sampleX = prevX + (bullet->x - prevX) * t;
			float sampleY = prevY + (bullet->y - prevY) * t;

			float normX = pixelToNormX(sampleX);
			float normY = pixelToNormY(sampleY);

			if (red_mode)
				addSource(densityTex, densityFBO, currentDensity, normX, normY, 1, 0, 0, 0.00008f);
			else
				addSource(densityTex, densityFBO, currentDensity, normX, normY, 0, 1, 0, 0.00008f);

			float actualVelX = (bullet->x - bullet->old_x) / DT;
			float actualVelY = (bullet->y - bullet->old_y) / DT;

			float normVelX = 0.1f * velPixelToNormX(actualVelX);
			float normVelY = 0.1f * velPixelToNormY(actualVelY);
			addSource(velocityTex, velocityFBO, currentVelocity, normX, normY, normVelX, normVelY, 0.0f, 0.00008f);
		}
	}



	GLuint clearColor[4] = { 0, 0, 0, 0 };
	glClearTexImage(obstacleTex, 0, GL_RGBA, GL_UNSIGNED_BYTE, clearColor);

	addObstacleStamp(protagonist.tex,
		static_cast<int>(protagonist.x), static_cast<int>(protagonist.y),
		protagonist.width, protagonist.height, true,
		1, true);

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
		if (enemy_ships[i]->tex != 0 && enemy_ships[i]->isOnscreen())
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













void display()
{
	// Fixed time step
	static double currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	static double accumulator = 0.0;

	double newTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	double frameTime = newTime - currentTime;
	currentTime = newTime;

	if (frameTime > DT * 10.0)
		frameTime = DT * 10.0;

	accumulator += frameTime;

	while (accumulator >= DT)
	{
		simulate();
		accumulator -= DT;
		GLOBAL_TIME += DT;
	}


	// Render to screen
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, windowWidth, windowHeight);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(displayProgram);
	setTextureUniform(displayProgram, "density", 0, densityTex[currentDensity]);
	setTextureUniform(displayProgram, "velocity", 1, velocityTex[currentVelocity]);
	setTextureUniform(displayProgram, "obstacles", 2, obstacleTex);
	setTextureUniform(displayProgram, "background", 3, background.tex);
	glUniform1f(glGetUniformLocation(displayProgram, "time"), GLOBAL_TIME);
	glUniform2f(glGetUniformLocation(displayProgram, "texelSize"), 1.0f / windowWidth, 1.0f / windowHeight);

	drawQuad();





	if (protagonist.tex != 0)
	{
		drawSprite(protagonist.tex,
			static_cast<int>(protagonist.x), static_cast<int>(protagonist.y),
			protagonist.width, protagonist.height, protagonist.under_fire);
	}

	for (size_t i = 0; i < foreground_chunked.size(); i++)
	{
		if (foreground_chunked[i].tex != 0 && foreground_chunked[i].isOnscreen())
		{
			drawSprite(foreground_chunked[i].tex,
				static_cast<int>(foreground_chunked[i].x), static_cast<int>(foreground_chunked[i].y),
				foreground_chunked[i].width, foreground_chunked[i].height, false);
		}
	}

	for (size_t i = 0; i < enemy_ships.size(); i++)
	{
		if (enemy_ships[i]->tex != 0 && enemy_ships[i]->isOnscreen())
		{
			drawSprite(enemy_ships[i]->tex,
				static_cast<int>(enemy_ships[i]->x), static_cast<int>(enemy_ships[i]->y),
				enemy_ships[i]->width, enemy_ships[i]->height, enemy_ships[i]->under_fire);
		}
	}



	displayFPS();



	//lines.clear();

	//for (size_t i = 0; i < ally_bullets.size(); i++)
	//{
	//	float vel_x = (ally_bullets[i]->x - ally_bullets[i]->old_x) / DT;
	//	float vel_y = (ally_bullets[i]->y - ally_bullets[i]->old_y) / DT;


	//	lines.push_back(Line(glm::vec2(ally_bullets[i]->x, ally_bullets[i]->y), glm::vec2(ally_bullets[i]->x + vel_x, ally_bullets[i]->y + vel_y), glm::vec4(1, 0, 0, 1)));

	//}


	//drawLinesWithWidth(lines, 4.0f);



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
			cout << "culling ally bullet" << endl;
			it = ally_bullets.erase(it);
		}
		else
			it++;
	}


	glutSwapBuffers();
	glutPostRedisplay();
}

void reshape(int w, int h) {
	windowWidth = w;
	windowHeight = h;
}

void keyboard(unsigned char key, int x, int y)
{
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

	case 27:  // ESC
	case 'q':
	case 'Q':
		exit(0);
		break;
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
	}
}

void specialKeys(int key, int x, int y) {
	if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
		shiftDown = true;
	}
}

void specialKeysUp(int key, int x, int y) {
	shiftDown = false;
}

void mouse(int button, int state, int x, int y) {
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

	protagonist.set_velocity(local_vel_x * windowWidth, local_vel_y * windowHeight);
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

	protagonist.set_velocity(local_vel_x * windowWidth, local_vel_y * windowHeight);
}

void keyboardup(unsigned char key, int x, int y) {
	switch (key) {
	case ' ': // Space bar

		spacePressed = false;




		break;
	}
}


#pragma comment(lib, "freeglut")
#pragma comment(lib, "glew32")








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



	initLineRenderer();



	initCollisionResources();

	// Load protagonist texture
	protagonist.tex = loadTextureFromFile_Triplet("media/protagonist_up.png", "media/protagonist_down.png", "media/protagonist_rest.png", &protagonist.width, &protagonist.height, protagonist.to_present_up_data, protagonist.to_present_down_data, protagonist.to_present_rest_data, protagonist);
	if (protagonist.tex == 0)
	{
		std::cout << "Warning: Could not load protagonist sprite" << std::endl;
		return 1;
	}

	protagonist.x = 200;
	protagonist.y = 300;

	background.tex = loadTextureFromFile("media/background.png", &background.width, &background.height, background.to_present_data);
	if (background.tex == 0)
	{
		std::cout << "Warning: Could not load background sprite" << std::endl;
		return 2;
	}

	if (!chunkForegroundTexture("media/foreground.png"))
	{
		std::cout << "Warning: Could not chunk foreground sprite" << std::endl;
		return 3;
	}

	bullet_template.tex = loadTextureFromFile("media/bullet.png", &bullet_template.width, &bullet_template.height, bullet_template.to_present_data);
	if (bullet_template.tex == 0)
	{
		std::cout << "Warning: Could not load bullet_template sprite" << std::endl;
		return 4;
	}

	enemy0_template.tex = loadTextureFromFile_Triplet("media/enemy0_up.png", "media/enemy0_down.png", "media/enemy0_rest.png", &enemy0_template.width, &enemy0_template.height, enemy0_template.to_present_up_data, enemy0_template.to_present_down_data, enemy0_template.to_present_rest_data, enemy0_template);
	if (enemy0_template.tex == 0)
	{
		std::cout << "Warning: Could not load enemy0_template sprite" << std::endl;
		return 5;
	}

	enemy1_template.tex = loadTextureFromFile_Triplet("media/enemy1_up.png", "media/enemy1_down.png", "media/enemy1_rest.png", &enemy1_template.width, &enemy1_template.height, enemy1_template.to_present_up_data, enemy1_template.to_present_down_data, enemy1_template.to_present_rest_data, enemy1_template);
	if (enemy1_template.tex == 0)
	{
		std::cout << "Warning: Could not load enemy1_template sprite" << std::endl;
		return 6;
	}

	enemy_ships.push_back(make_unique<enemy_ship>(enemy0_template));
	enemy_ships.push_back(make_unique<enemy_ship>(enemy1_template));

	enemy_ships[0]->x = 300;
	enemy_ships[0]->y = 200;
	enemy_ships[0]->manually_update_data(enemy0_template.to_present_up_data, enemy0_template.to_present_down_data, enemy0_template.to_present_rest_data);

	enemy_ships[1]->x = 400;
	enemy_ships[1]->y = 300;
	enemy_ships[1]->manually_update_data(enemy1_template.to_present_up_data, enemy1_template.to_present_down_data, enemy1_template.to_present_rest_data);

	//    printControls();

		// Register callbacks
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialKeys);
	glutSpecialUpFunc(specialKeysUp);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(passiveMotion);

	glutSpecialFunc(specialKeyboard);
	glutSpecialUpFunc(specialKeyboardUp);
	glutKeyboardUpFunc(keyboardup);

	glutFullScreen();

	// Main loop
	glutMainLoop();

	return 0;
}
