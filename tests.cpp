#include "gtest/gtest.h"

#include "main.h"

TEST(Test, Test_foofail)
{
	std::optional<int> opt; 
	if (opt) //эту штуку сокращаю ниже
	{
		std::optional<int> b = cat(*opt); //Если opt не False, то использовать в качестве аргумента функции cat
	}

	monadic_optional<int> opt2; //Объект, у которого есть метод and_then
	monadic_optional<int> t = opt2.and_then(cat); //Здесь ошибка, error_type для cat, и для cat не удается определить шаблон
	
	
	
	
	//EXPECT_EQ(b.foo(10), 10);
}
