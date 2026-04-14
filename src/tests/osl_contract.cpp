#include "tests/gtest/gtest.h"

#include "textures/osl.h"

using namespace pbrt;

TEST(OSLContract, RequiresShaderOrGroupSpec) {
    OSLShaderConfig config;
    config.outputName = "result";
    EXPECT_FALSE(ValidateAndNormalizeOSLShaderConfig(&config, "test"));
}

TEST(OSLContract, GroupspecWinsOverShader) {
    OSLShaderConfig config;
    config.shader = "foo";
    config.groupSpec = "shader foo foo_layer";
    config.outputName = "result";
    EXPECT_TRUE(ValidateAndNormalizeOSLShaderConfig(&config, "test"));
    EXPECT_EQ("", config.shader);
}

TEST(OSLContract, EmptyOutputGetsDefaulted) {
    OSLShaderConfig config;
    config.shader = "foo";
    EXPECT_TRUE(ValidateAndNormalizeOSLShaderConfig(&config, "test"));
    EXPECT_EQ("result", config.outputName);
}

TEST(OSLContract, LayerWithoutShaderIsRejected) {
    OSLShaderConfig config;
    config.layer = "layer0";
    config.outputName = "result";
    EXPECT_FALSE(ValidateAndNormalizeOSLShaderConfig(&config, "test"));
}

#ifdef PBRT_ENABLE_OSL
TEST(OSLContract, RuntimeCountersReadable) {
    OSLRuntimeCounters counters = GetOSLRuntimeCounters();
    EXPECT_GE(counters.executions, 0u);
    EXPECT_GE(counters.groupCacheHits, 0u);
    EXPECT_GE(counters.fallbackCount, 0u);
}

TEST(OSLContract, UnknownClosureLookupFails) {
    EXPECT_EQ(-1, GetOSLClosureIdByName("pbrt_unknown_closure"));
}
#endif
