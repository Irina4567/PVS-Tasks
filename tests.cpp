#include "gtest/gtest.h"

#include "main.h"
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
TEST(Test, Test_foofail)
{
	auto i = stoi("123"); //преобразует строку в число, если 

	if (i)
	{
		auto s = to_string(i);
	}
	std::optional<std::string> opt_string{ "fds" };
	if (opt_string) {
		std::size_t s = opt_string->size();
	}


	optional<string> o{ "234" };

	std::optional<int> io = o.and_then(stoi);

	//monadic_optional<string> opt{"123"};
	//


	//EXPECT_EQ(r, 2);
}

//TEST(Test, Test_sort1)
//{
//	list<int> a{ 1, 3, 2 };
//	sort(a);
//	//EXPECT_EQ(sort(a), {1, 2, 3});
//}