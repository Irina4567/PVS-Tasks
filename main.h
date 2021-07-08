#include <optional>
#ifndef MAIN_H
#define MAIN_H

template <class T>
std::optional<T> cat(T c)
{
	retutn(std::optional<T>)c;
}


template <class T> 
class monadic_optional : public std::optional
{
public:
	template <class T2>
	std::optional<T2> and_then(function<std::optional<T2>(T)> f)
	{
		if (this) {
			return (std::optional)f(*this);
		}
	}
};
#endif // MAIN_H