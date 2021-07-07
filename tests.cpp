#include "gtest/gtest.h"

#include "main.h"

TEST(Test, Test_foofail)
{
	monadic_optional<int> b;
	EXPECT_EQ(b.foo(10), 10);
}