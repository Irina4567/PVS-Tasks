#include "gtest/gtest.h"

#include "main.h"

TEST(Test, Test_foofail)
{
	std::optional<int> opt;
	if (opt)
	{
		std::optional<int> b = cat(*opt);
	}

	monadic_optional<int> opt2;
	monadic_optional<int> t = opt2.and_then(cat);
	
	
	
	
	//EXPECT_EQ(b.foo(10), 10);
}