#pragma once
#include <iostream>
#include <cmath>
#include <cstring>

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "    FAIL " << __FILE__ << ":" << __LINE__ << "  — " #cond "\n"; \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps)  ASSERT(std::abs(static_cast<float>(a) - static_cast<float>(b)) < (eps))
#define ASSERT_EQ(a, b)         ASSERT((a) == (b))
#define ASSERT_NE(a, b)         ASSERT((a) != (b))
#define ASSERT_GT(a, b)         ASSERT((a) >  (b))
#define ASSERT_LT(a, b)         ASSERT((a) <  (b))
#define ASSERT_GE(a, b)         ASSERT((a) >= (b))
#define ASSERT_LE(a, b)         ASSERT((a) <= (b))
#define ASSERT_FINITE(v)        ASSERT(std::isfinite(static_cast<float>(v)))

#define TEST(name) static void test_##name()

#define RUN_TEST(name) do { \
    int _before = g_fail; \
    std::cout << "  " #name " ... " << std::flush; \
    test_##name(); \
    std::cout << (g_fail == _before ? "ok" : "FAILED") << "\n"; \
} while(0)

inline int report(const char* suite) {
    int total = g_pass + g_fail;
    std::cout << "\n[" << suite << "] " << g_pass << "/" << total << " passed";
    if (g_fail > 0) std::cout << "  (" << g_fail << " FAILED)";
    std::cout << "\n";
    return g_fail > 0 ? 1 : 0;
}
