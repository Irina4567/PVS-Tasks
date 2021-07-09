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



//template<class T>
//void sort(T&& vec, bool(*)(T) pred)
//{
//	std::sort(vec.begin(), vec.end());
//}



#endif // MAIN_H