#include <optional>
#ifndef MAIN_H
#define MAIN_H

template <class T>
class monadic_optional : public std::optional<T>
{
public:
	T foo(T a) {
		return a;
	};
};

#endif // MAIN_H