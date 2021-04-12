#if 1
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <iostream>

struct MyListener : Catch::TestEventListenerBase {

	using TestEventListenerBase::TestEventListenerBase; // inherit constructor

	virtual void testCaseStarting(Catch::TestCaseInfo const& testInfo) override {
		std::cout << "Execute " << testInfo.tagsAsString() << " " << testInfo.name << std::endl;
	}
};
CATCH_REGISTER_LISTENER(MyListener)

#elif 1

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
void testIntegrationBounds();
int main()
{
	testIntegrationBounds();
	return 0;
}

#elif 1

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
void testMarchingCubes();
int main()
{
	testMarchingCubes();
	return 0;
}

#endif
