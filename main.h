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
>>>>>>> b07b037b92d0356dfa87ecbf60b0da3f652e1e8e
	{
		//std::invoke - ��������������� ������� ������� �������� (� �.�. ������), ��������� �� ������� � ��������� �� �������-����� �������
		//(obj->*mem_fn_ptr)( args... );

		//std::forward<T>(t) - ��������� ������������ ��� ���������, t - �������� ������� � ����� T&& - ������������� ������. 
		//����� ������ ������������ ������ &&, �� ���� ��� rvalue("��������� ������") ��� ��� lvalue(������������� ������)
		//std::forward ��������� ����� ������������� ������ � ������������� ��� � rvalue ������ ���� ��� ������� ��������
		//����� - std::forward ��������� ��������� ��� ����������� ��������� ��� �������� ��� ������ �������

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
