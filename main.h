#include <optional>
#ifndef MAIN_H
#define MAIN_H

template <class T>
std::optional<T> cat(T c) //Тестовая функция cat
{
	retutn(std::optional<T>)c; // возвращает аргумент, преобразованный к типу std::optional<T>
}


template <class T> 
class monadic_optional : public std::optional //наследование от std::optional
{
public:
	template <class T2>
	std::optional<T2> and_then(function<std::optional<T2>(T)> f) 
	//ans_then принимает любую функцию, которая возвращает std::optional<T2>. 
	//Если объект не пуст, то функция вызывается со значением объекта и and_then возвращает std::optional
	{
		if (this) {
			return (std::optional)f(*this);
		}
	}
};
#endif // MAIN_H
