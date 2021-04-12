#include <iostream>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/IconsFontAwesome5.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuMat/src/Context.h>

#include "visualizer.h"
#include <opengl_renderer.h>

#include <cmrc/cmrc.hpp>

CMRC_DECLARE(ui);

int mainWindow();
int demoWindow();
int testDDA();
int testCubic();

int main(int, char**)
{
	return mainWindow();
	//return demoWindow()
	//return testDDA();
	//return testScriptLoading();
	//return testInpainting();
	//return testCubic();
}

//IMPLEMENTATION


static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

static const std::map<GLenum, const char*> MESSAGE_SOURCES = {
	{GL_DEBUG_SOURCE_API, "API"},
	{GL_DEBUG_SOURCE_WINDOW_SYSTEM, "Window"},
	{GL_DEBUG_SOURCE_SHADER_COMPILER, "Shader-Compiler"},
	{GL_DEBUG_SOURCE_THIRD_PARTY, "Third-Party"},
};
static const std::map<GLenum, const char*> MESSAGE_TYPES = {
	{GL_DEBUG_TYPE_ERROR, "ERROR"},
	{GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, "deprecation"},
	{GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR, "undefined-behaviour"},
	{GL_DEBUG_TYPE_PORTABILITY, "portability"},
	{GL_DEBUG_TYPE_PERFORMANCE, "performance"},
};
static const std::map<GLenum, const char*> MESSAGE_SEVERITY = {
	{GL_DEBUG_SEVERITY_HIGH, "HIGH"},
	{GL_DEBUG_SEVERITY_MEDIUM, "medium"},
	{GL_DEBUG_SEVERITY_LOW, "low"},
	{GL_DEBUG_SEVERITY_NOTIFICATION, "info"},
};
static const char* MESSAGE_UNKNOWN = "unknown";
template <template<class, class, class...> class C, typename K, typename V, typename... Args>
V getDefault(const C<K, V, Args...>& m, K const& key, const V& defval)
{
	typename C<K, V, Args...>::const_iterator it = m.find(key);
	if (it == m.end())
		return defval;
	return it->second;
}
void GLAPIENTRY
MessageCallback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam)
{
	if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
	if (type == GL_DEBUG_TYPE_PERFORMANCE) return;
	const char* sourceStr = getDefault(MESSAGE_SOURCES, source, MESSAGE_UNKNOWN);
	const char* typeStr = getDefault(MESSAGE_TYPES, type, MESSAGE_UNKNOWN);
	const char* severityStr = getDefault(MESSAGE_SEVERITY, severity, MESSAGE_UNKNOWN);
	fprintf(stderr, "GL CALLBACK: source: %s, type: %s (0x%x), severity: %s (0x%x), message: %s\n",
		sourceStr, typeStr, type, severityStr, severity, message);
}

int mainWindow()
{
	//Registrations
	renderer::OpenGLRasterization::Register();
	
	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;

	// Decide GL+GLSL versions
#if __APPLE__
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 4.3 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif
#if !defined(NDEBUG)
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

	// Create window with graphics context
	GLFWwindow* window = glfwCreateWindow(1420, 860, "Basic Volume Renderer", NULL, NULL);
	if (window == NULL)
		return 1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	// Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
	bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
	bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
	bool err = gladLoadGL() == 0;
#else
	bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return 1;
	}

#if !defined(NDEBUG)
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(MessageCallback, 0);
#endif

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
	// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Read 'misc/fonts/README.txt' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != NULL);

	// Add icons
	io.Fonts->AddFontDefault();
	static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
	ImFontConfig icons_config; icons_config.MergeMode = true; icons_config.PixelSnapH = true;
	auto fs = cmrc::ui::get_filesystem();
	auto fontMemFile = fs.open("resources/" FONT_ICON_FILE_NAME_FAS);
	auto* fontMemory = new unsigned char[fontMemFile.size()];
	memcpy(fontMemory, fontMemFile.begin(), fontMemFile.size());
	io.Fonts->AddFontFromMemoryTTF(fontMemory, int(fontMemFile.size()), 16.0f, &icons_config, icons_ranges);

	std::unique_ptr<Visualizer> vis = std::make_unique<Visualizer>(window);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		// Poll and handle events (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		glfwPollEvents();

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// show UI windows
		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(300, 680), ImGuiCond_FirstUseEver);
		ImGui::Begin("Main", nullptr, ImGuiWindowFlags_MenuBar);
		vis->specifyUI();
		ImGui::End();

		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(vis->clearColor_.x, vis->clearColor_.y, vis->clearColor_.z, vis->clearColor_.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		vis->render(display_w, display_h);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	vis.reset();
	cuMat::Context::current().destroy();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

int demoWindow()
{
	// DEMO:
		// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		return 1;

	// Decide GL+GLSL versions
#if __APPLE__
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

	// Create window with graphics context
	GLFWwindow* window = glfwCreateWindow(1420, 860, "Isosurface super resolution", NULL, NULL);
	if (window == NULL)
		return 1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	// Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
	bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
	bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
	bool err = gladLoadGL() == 0;
#else
	bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return 1;
	}

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Our state
	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		// Poll and handle events (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		glfwPollEvents();

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
		if (show_demo_window)
			ImGui::ShowDemoWindow(&show_demo_window);

		// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
		{
			static float f = 0.0f;
			static int counter = 0;

			ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

			ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
			ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
			ImGui::Checkbox("Another Window", &show_another_window);

			ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
			ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

			if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
				counter++;
			ImGui::SameLine();
			ImGui::Text("counter = %d", counter);

			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();
		}

		// 3. Show another simple window.
		if (show_another_window)
		{
			ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
			ImGui::Text("Hello from another window!");
			if (ImGui::Button("Close Me"))
				show_another_window = false;
			ImGui::End();
		}

		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

#include <helper_math.cuh>
struct DDA
{
private:
	int3 pStep;
	float3 tDel, tSide;
public:
	/**
	 * The time since entering the grid
	 */
	float t;
	/**
	 * The current voxel position
	 */
	int3 p;

	/**
	 * \brief Prepares the DDA traversal.
	 *   the variable 'p' will contain the voxel position of the entry
	 * \param entry the entry position into the volume
	 * \param dir the ray direction
	 * \param voxelSize the size of the voxels
	 */
	__host__ __device__ DDA(const float3& entry, const float3 dir, const float3& voxelSize)
	{
		float3 pStepF = fsignf(dir);
		float3 pF = entry / voxelSize;
		tDel = fabs(voxelSize / dir);
		tSide = ((floorf(pF) - pF + 0.5)*pStepF + 0.5) * tDel;
		pStep = make_int3(pStepF);
		p = make_int3(floorf(pF));
		t = 0;
	}

	/**
	 * Steps into the next voxel along the ray. The variables 'p' and 't' are updated
	 */
	__host__ __device__ void step()
	{
		int3 mask;
		mask.x = (tSide.x < tSide.y) & (tSide.x <= tSide.z);
		mask.y = (tSide.y < tSide.z) & (tSide.y <= tSide.x);
		mask.z = (tSide.z < tSide.x) & (tSide.z <= tSide.y);
		t = mask.x ? tSide.x : (mask.y ? tSide.y : tSide.z);
		tSide += make_float3(mask) * tDel;
		p += mask * pStep;
	}
};
int testDDA()
{
	float3 ray_start = make_float3(0.5,0.2,2.3);
	float3 ray_end = make_float3(-0.6, 0.7, -1.4);
	float voxelSize = 1.0f / 8.0f;
	float3 ray_dir = normalize(ray_end - ray_start);
	float ray_length = length(ray_end - ray_start);
	std::cout << "Voxel size: " << voxelSize << std::endl;
	std::cout << "Starting position: " << ray_start.x << " " << ray_start.y << " " << ray_start.z << std::endl;
	std::cout << "Ending position: " << ray_end.x << " " << ray_end.y << " " << ray_end.z << std::endl;
	std::cout << "Ray length: " << ray_length << std::endl;
	std::cout << "Voxel ID's from start to end:" << std::endl;
	DDA dda(ray_start, ray_dir, make_float3(voxelSize));
	while(true)
	{
		float time = dda.t;
		int3 posI = dda.p;
		if (time > ray_length) break;

		dda.step();
		float timeOut = dda.t;

		float3 posF = ray_start + ray_dir * time;
		float3 entry = (ray_start + ray_dir * time) / make_float3(voxelSize) - make_float3(posI);
		float3 exit = entry + ray_dir * (timeOut - time);
		std::cout << "pos: (" << posI.x << " " << posI.y << " " << posI.z <<
			"), t: " << time <<
			", posF: (" << posF.x << " " << posF.y << " " << posF.z << ")" <<
			", entry: (" << entry.x << " " << entry.y << " " << entry.z << ")" <<
			", exit: (" << exit.x << " " << exit.y << " " << exit.z << ")\n";
	}
	return 0;
}


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
struct CubicPolynomial
{
	static constexpr float EPS = 1e-5;
	/**
	 * Returns the cubic polynomial factors for the ray r(t)=entry+t*dir
	 * traversing the voxel with corner values given by 'vals' and accessing the
	 * tri-linear interpolated values.
	 */
	static __host__ __device__ float4 getFactors(
		const float vals[8], const float3& entry, const float3& dir)
	{
		const float v0 = vals[0], v1 = vals[1], v2 = vals[2], v3 = vals[3];
		const float v4 = vals[4], v5 = vals[5], v6 = vals[6], v7 = vals[7];
		const float ex = entry.x, ey = entry.y, ez = entry.z;
		const float dx = dir.x, dy = dir.y, dz = dir.z;
		//TODO: simplify
		const float a = -dx * dy*dz*v0 + dx * dy*dz*v1 + dx * dy*dz*v2 - dx * dy*dz*v3 + dx * dy*dz*v4 - dx * dy*dz*v5 - dx * dy*dz*v6 + dx * dy*dz*v7;
		const float b = -dx * dy*ez*v0 + dx * dy*ez*v1 + dx * dy*ez*v2 - dx * dy*ez*v3 + dx * dy*ez*v4 - dx * dy*ez*v5 - dx * dy*ez*v6 + dx * dy*ez*v7 + dx * dy*v0 - dx * dy*v1 - dx * dy*v2 + dx * dy*v3 - dx * dz*ey*v0 + dx * dz*ey*v1 + dx * dz*ey*v2 - dx * dz*ey*v3 + dx * dz*ey*v4 - dx * dz*ey*v5 - dx * dz*ey*v6 + dx * dz*ey*v7 + dx * dz*v0 - dx * dz*v1 - dx * dz*v4 + dx * dz*v5 - dy * dz*ex*v0 + dy * dz*ex*v1 + dy * dz*ex*v2 - dy * dz*ex*v3 + dy * dz*ex*v4 - dy * dz*ex*v5 - dy * dz*ex*v6 + dy * dz*ex*v7 + dy * dz*v0 - dy * dz*v2 - dy * dz*v4 + dy * dz*v6;
		const float c = -dx * ey*ez*v0 + dx * ey*ez*v1 + dx * ey*ez*v2 - dx * ey*ez*v3 + dx * ey*ez*v4 - dx * ey*ez*v5 - dx * ey*ez*v6 + dx * ey*ez*v7 + dx * ey*v0 - dx * ey*v1 - dx * ey*v2 + dx * ey*v3 + dx * ez*v0 - dx * ez*v1 - dx * ez*v4 + dx * ez*v5 - dx * v0 + dx * v1 - dy * ex*ez*v0 + dy * ex*ez*v1 + dy * ex*ez*v2 - dy * ex*ez*v3 + dy * ex*ez*v4 - dy * ex*ez*v5 - dy * ex*ez*v6 + dy * ex*ez*v7 + dy * ex*v0 - dy * ex*v1 - dy * ex*v2 + dy * ex*v3 + dy * ez*v0 - dy * ez*v2 - dy * ez*v4 + dy * ez*v6 - dy * v0 + dy * v2 - dz * ex*ey*v0 + dz * ex*ey*v1 + dz * ex*ey*v2 - dz * ex*ey*v3 + dz * ex*ey*v4 - dz * ex*ey*v5 - dz * ex*ey*v6 + dz * ex*ey*v7 + dz * ex*v0 - dz * ex*v1 - dz * ex*v4 + dz * ex*v5 + dz * ey*v0 - dz * ey*v2 - dz * ey*v4 + dz * ey*v6 - dz * v0 + dz * v4;
		const float d = -ex * ey*ez*v0 + ex * ey*ez*v1 + ex * ey*ez*v2 - ex * ey*ez*v3 + ex * ey*ez*v4 - ex * ey*ez*v5 - ex * ey*ez*v6 + ex * ey*ez*v7 + ex * ey*v0 - ex * ey*v1 - ex * ey*v2 + ex * ey*v3 + ex * ez*v0 - ex * ez*v1 - ex * ez*v4 + ex * ez*v5 - ex * v0 + ex * v1 + ey * ez*v0 - ey * ez*v2 - ey * ez*v4 + ey * ez*v6 - ey * v0 + ey * v2 - ez * v0 + ez * v4 + v0;
		return make_float4(a, b, c, d);
	}

	static __host__ __device__ float evalCubic(const float4& factors, float t)
	{
		return factors.w + t * (factors.z + t * (factors.y + t * factors.x));
	}

	/**
	 * \brief Computes the roots of the cubic using the analytic hyperbolic equations.
	 * \param factors the factors of the polynomial f(x)=ax^3+bx^2+cx+d = 0.
	 * \param roots will be filled with the values of 'x' at the roots
	 * \return the number of real roots, 0, 1, 3
	 */
	static __host__ __device__ int rootsHyperbolic(const float4& factors, float roots[3])
	{
		//extract factors
		const double a = factors.x, b = factors.y, c = factors.z, d = factors.w;

		//convert to depressed cubic t^3+pt+q=0
		const double p = (3 * a*c - b * b) / (3 * a*a);
		const double q = (2 * b*b*b - 9 * a*b*c + 27 * a*a*d) / (27 * a*a*a);

#define t2x(t) ((t)-b/(3*a))
#define safeSqrt(x) sqrt(fmax(0.0, (x)))
#define safeAcos(x) acos(fmax(-1.0, fmin(1.0, (x))))

		if (abs(p) < EPS)
		{
			//there exists exactly one root
			roots[0] = t2x(cbrt(-q));
			return 1;
		}
		//formular of Francois Viète
		//https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_solution_for_three_real_roots
		const double Delta = 4.0f * p*p*p + 27.0f * q*q;
		if (Delta > 0)
		{
			//one real root
			double t0;
			if (p < 0)
				t0 = -2.0f * fsign(q)*sqrt(-p / 3.0f)*cosh(1.0f / 3 * acosh(-3.0f * abs(q) / (2.0f * p)*sqrt(-3.0f / p)));
			else
				t0 = -2.0f * sqrt(p / 3.0f)*sinh(1.0f / 3.0f * asinh(3.0f * q / (2.0f * p)*sqrt(3.0f / p)));
			roots[0] = t2x(t0);
			return 1;
		}
		else
		{
			//three real roots
			const double f1 = 2.0f * safeSqrt(-p / 3.0f);
			const double f2 = 1.0f / 3.0f * safeAcos(3.0f * q / (2.0f * p)*safeSqrt(-3.0f / p));
			for (int k = 0; k < 3; ++k) {
				float t = f1 * cos(f2 - 2.0f * M_PI*k / 3.0f);
				float vtest = t * t*t + p * t + q;
				roots[k] = t2x(t);
			}
			return 3;
		}

#undef safeAcos
#undef safeSqrt
#undef t2x
	}
};
static float lerp3D(const float vals[8], const float3& p)
{
	return lerp(
		lerp(lerp(vals[0], vals[1], p.x),
			lerp(vals[2], vals[3], p.x),
			p.y),
		lerp(lerp(vals[4], vals[5], p.x),
			lerp(vals[6], vals[7], p.x),
			p.y),
		p.z);
}
int testCubic()
{
	std::default_random_engine rnd;
	std::uniform_real_distribution<float> distr(0, 1);
	
	float vals[8];
	std::cout << "Vertex values:";
	for (int i=0; i<8; ++i)
	{
		vals[i] = distr(rnd)*2-1;
		std::cout << " " << vals[i];
	}
	std::cout << std::endl;

	float3 entry = make_float3(distr(rnd), distr(rnd), distr(rnd));
	float3 exit = make_float3(distr(rnd), distr(rnd), distr(rnd));
	float3 dir = exit - entry;
	std::cout << "Entry: " << entry.x << " " << entry.y << " " << entry.z << std::endl;
	std::cout << "Exit: " << exit.x << " " << exit.y << " " << exit.z << std::endl;
	std::cout << "Direction: " << dir.x << " " << dir.y << " " << dir.z << std::endl;

	float4 factors = CubicPolynomial::getFactors(vals, entry, dir);
	std::cout << std::endl;
	std::cout << "Cubic equation:\n";
	std::cout << factors.x << " * x^3 + " << factors.y << " * x^2 + " <<
		factors.z << " * x + " << factors.w << std::endl;

	std::cout << std::endl;
	std::cout << "Check, if the cubic factors are correct:" << std::endl;
	const double a = factors.x, b = factors.y, c = factors.z, d = factors.w;
	const double p = (3 * a*c - b * b) / (3 * a*a);
	const double q = (2 * b*b*b - 9 * a*b*c + 27 * a*a*d) / (27 * a*a*a);
	for (float t=0; t<=1; t += 1/8.0f)  // NOLINT(cert-flp30-c)
	{
		float v1 = lerp3D(vals, entry + t * dir);
		float v2 = CubicPolynomial::evalCubic(factors, t);
		float t2 = t + b / (3 * a);
		float v3 = (t2*t2*t2 + p*t2 + q) * a;
		std::cout << "t=" << t << ", v1=" << v1 << ", v2=" << v2 << ", v3=" << v3 << std::endl;
	}

	//compute roots
	float roots[3];
	int numRoots = CubicPolynomial::rootsHyperbolic(factors, roots);
	std::cout << "Number of roots: " << numRoots << "\n";
	for (int k=0; k<numRoots; ++k)
	{
		std::cout << "  t=" << roots[k] << ", v(t)=" << CubicPolynomial::evalCubic(factors, roots[k]) << std::endl;
	}

	//now test some simple cubics
	auto testCubic = [](float a, float b, float c, float d)
	{
		float roots[3];
		float4 factors = make_float4(a, b, c, d);
		int numRoots = CubicPolynomial::rootsHyperbolic(factors, roots);
		std::cout << "\nCubic equation " << a << "*x^3 + " << b << "*x^2 + " << c << "*x + " << d << " = 0\n";
		for (int k = 0; k < numRoots; ++k)
		{
			std::cout << "  t=" << roots[k] << ", v(t)=" << CubicPolynomial::evalCubic(factors, roots[k]) << std::endl;
		}
	};
	testCubic(1, 0, 1, 0);
	testCubic(1, 1.5, -1, -0.5);
	testCubic(2, -0.5, 2, -1);
	testCubic(2, -0.5, 2, +1);
	testCubic(1, 3, -6, -8);
	return 0;
}
