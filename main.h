#include <optional>
#ifndef MAIN_H
#define MAIN_H

template <class T> //шаблон класса, так как надо чтобы для разных вариантов робило
class monadic_optional : public std::optional<T> 
	//открытое наследование от шаблона класса std::optional, private члены из std::optional недоступны в monadic_optional, public->public, protected->protected
{
public:
	T foo(T a) {
		return a;
	};
};

#endif // MAIN_H