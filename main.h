#ifndef MAIN_H
#define MAIN_H
#include <optional>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;

<<<<<<< HEAD
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
=======
template <class T>
std::optional<T> cat(T c) //РўРµСЃС‚РѕРІР°СЏ С„СѓРЅРєС†РёСЏ cat
{
	retutn(std::optional<T>)c; // РІРѕР·РІСЂР°С‰Р°РµС‚ Р°СЂРіСѓРјРµРЅС‚, РїСЂРµРѕР±СЂР°Р·РѕРІР°РЅРЅС‹Р№ Рє С‚РёРїСѓ std::optional<T>
}


template <class T> 
class monadic_optional : public std::optional //РЅР°СЃР»РµРґРѕРІР°РЅРёРµ РѕС‚ std::optional
{
public:
	template <class T2>
	std::optional<T2> and_then(function<std::optional<T2>(T)> f) 
	//ans_then РїСЂРёРЅРёРјР°РµС‚ Р»СЋР±СѓСЋ С„СѓРЅРєС†РёСЋ, РєРѕС‚РѕСЂР°СЏ РІРѕР·РІСЂР°С‰Р°РµС‚ std::optional<T2>. 
	//Р•СЃР»Рё РѕР±СЉРµРєС‚ РЅРµ РїСѓСЃС‚, С‚Рѕ С„СѓРЅРєС†РёСЏ РІС‹Р·С‹РІР°РµС‚СЃСЏ СЃРѕ Р·РЅР°С‡РµРЅРёРµРј РѕР±СЉРµРєС‚Р° Рё and_then РІРѕР·РІСЂР°С‰Р°РµС‚ std::optional
>>>>>>> b07b037b92d0356dfa87ecbf60b0da3f652e1e8e
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
<<<<<<< HEAD



//template<class T>
//void sort(T&& vec, bool(*)(T) pred)
//{
//	std::sort(vec.begin(), vec.end());
//}



#endif // MAIN_H
=======
#endif // MAIN_H
>>>>>>> b07b037b92d0356dfa87ecbf60b0da3f652e1e8e
