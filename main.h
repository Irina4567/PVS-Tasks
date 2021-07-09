#ifndef MAIN_H
#define MAIN_H
#include <optional>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;

//template <class T>
//std::optional<T> cat(T c)
//{
//	cout << c << " c" << endl;
//	//cout << *c << " *c" << endl;
//	cout << *(std::optional<T>)c << " (std::optional<T>)c" << endl;
//
//	return (std::optional<T>)c;
//}

template <class T> 
class optional : public std:optional<T>
{
public:
	T t;
	template <class T>
	constexpr auto and_then(F&& f)
	{
		//std::invoke - унифицированным образом вызвает функторы (в т.ч. лямбды), указатели на функции и указатели на функции-члены классов
		//(obj->*mem_fn_ptr)( args... );

		//std::forward<T>(t) - сохраняет оригинальный тип аргумента, t - аргумент функции с типом T&& - универсальная ссылка. 
		//можно двояко воспринимать ссылки &&, то есть как rvalue("временные ссылки") или как lvalue(универсальная ссылка)
		//std::forward позволяет взять универсальную ссылку и преобразовать его в rvalue только если оно таковым является
		//проще - std::forward позволяет сохранить тип вызывающего агрумента при передаче его другой функции

		if (*this->has_value()) {
			return std::invoke(std::forward<F>(f), **this);
		}
		else
		{
			return std::nullopt;
		}
	}
};



//template<class T>
//void sort(T&& vec, bool(*)(T) pred)
//{
//	std::sort(vec.begin(), vec.end());
//}



#endif // MAIN_H