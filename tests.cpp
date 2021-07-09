#include "gtest/gtest.h"

#include "main.h"
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;
TEST(Test, Test_foofail)
{
<<<<<<< HEAD
	auto i = stoi("123"); //����������� ������ � �����, ���� 

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
	if (opt) //эту штуку сокращаю ниже
	{
		std::optional<int> b = cat(*opt); //Если opt не False, то использовать в качестве аргумента функции cat
	}

	monadic_optional<int> opt2; //Объект, у которого есть метод and_then
	monadic_optional<int> t = opt2.and_then(cat); //Здесь ошибка, error_type для cat, и для cat не удается определить шаблон
	
	
	
	
	//EXPECT_EQ(b.foo(10), 10);
}
>>>>>>> b07b037b92d0356dfa87ecbf60b0da3f652e1e8e
