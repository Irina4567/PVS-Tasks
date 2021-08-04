#include "gtest/gtest.h"

#include "main.h"
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
TEST(Test, Test_foofail)
{
<<<<<<< HEAD
	auto i = stoi("123"); //ïðåîáðàçóåò ñòðîêó â ÷èñëî, åñëè 

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
=======
	std::optional<int> opt; 
	if (opt) //ÑÑ‚Ñƒ ÑˆÑ‚ÑƒÐºÑƒ ÑÐ¾ÐºÑ€Ð°Ñ‰Ð°ÑŽ Ð½Ð¸Ð¶Ðµ
	{
		std::optional<int> b = cat(*opt); //Ð•ÑÐ»Ð¸ opt Ð½Ðµ False, Ñ‚Ð¾ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÑŒ Ð² ÐºÐ°Ñ‡ÐµÑÑ‚Ð²Ðµ Ð°Ñ€Ð³ÑƒÐ¼ÐµÐ½Ñ‚Ð° Ñ„ÑƒÐ½ÐºÑ†Ð¸Ð¸ cat
	}

	monadic_optional<int> opt2; //ÐžÐ±ÑŠÐµÐºÑ‚, Ñƒ ÐºÐ¾Ñ‚Ð¾Ñ€Ð¾Ð³Ð¾ ÐµÑÑ‚ÑŒ Ð¼ÐµÑ‚Ð¾Ð´ and_then
	monadic_optional<int> t = opt2.and_then(cat); //Ð—Ð´ÐµÑÑŒ Ð¾ÑˆÐ¸Ð±ÐºÐ°, error_type Ð´Ð»Ñ cat, Ð¸ Ð´Ð»Ñ cat Ð½Ðµ ÑƒÐ´Ð°ÐµÑ‚ÑÑ Ð¾Ð¿Ñ€ÐµÐ´ÐµÐ»Ð¸Ñ‚ÑŒ ÑˆÐ°Ð±Ð»Ð¾Ð½
	
	
	
	
	//EXPECT_EQ(b.foo(10), 10);
}
>>>>>>> b07b037b92d0356dfa87ecbf60b0da3f652e1e8e

