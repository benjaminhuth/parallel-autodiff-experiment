#ifndef AUTODIFF_TUPLE_HPP_INCLUDED
#define AUTODIFF_TUPLE_HPP_INCLUDED

#include <array>

template<std::size_t N>
struct autodiff_tuple
{
    std::array<double, N> array;
    
    autodiff_tuple() = default;
    
    autodiff_tuple &operator=(const autodiff_tuple &) = default; 	
    
    autodiff_tuple(double a) {
        for(std::size_t i=0; i<N; ++i)
            array[i] = a;
    }
    
    template<std::size_t M>
    void set_to_one() {   
        static_assert(M < N, "access error");
        for(std::size_t i=0; i<N; ++i)
            if( i == M )
                array[i] = 1.0;
            else
                array[i] = 0.0;
    }
    
    // OPERATOR *=
    autodiff_tuple& operator*=(const autodiff_tuple& rhs)
    {
        for(std::size_t i=0; i<N; ++i)
            array[i] = array[i] * rhs.array[i];
        
        return *this;
    }
    
    autodiff_tuple& operator*=(double rhs)
    {
        for(std::size_t i=0; i<N; ++i)
            array[i] = array[i] * rhs;
        
        return *this;
    }
    
    // OPERATOR +=
    autodiff_tuple& operator+=(const autodiff_tuple& rhs)
    {
        for(std::size_t i=0; i<N; ++i)
            array[i] = array[i] + rhs.array[i];
        
        return *this;
    }
    
    autodiff_tuple& operator+=(double rhs)
    {
        for(std::size_t i=0; i<N; ++i)
            array[i] = array[i] + rhs;
        
        return *this;
    }
    
    
    
};

// OPERATOR *
template<std::size_t N>
auto operator*(const autodiff_tuple<N> &lhs, const autodiff_tuple<N> &rhs) {
    autodiff_tuple<N> ret;
    for(std::size_t i=0; i<N; ++i)
        ret.array[i] = lhs.array[i] * rhs.array[i];
    return ret;
}

template<std::size_t N>
auto operator*(const autodiff_tuple<N> &lhs, double rhs) {
    autodiff_tuple<N> ret;
    for(std::size_t i=0; i<N; ++i)
        ret.array[i] = lhs.array[i] * rhs;
    return ret;
}

template<std::size_t N>
auto operator*(double lhs, const autodiff_tuple<N> &rhs) {
    return rhs * lhs;
}

// OPERATOR +
template<std::size_t N>
auto operator+(const autodiff_tuple<N> &lhs, const autodiff_tuple<N> &rhs) {
    autodiff_tuple<N> ret;
    for(std::size_t i=0; i<N; ++i)
        ret.array[i] = lhs.array[i] + rhs.array[i];
    return ret;
}

template<std::size_t N>
auto operator+(const autodiff_tuple<N> &lhs, double rhs) {
    autodiff_tuple<N> ret;
    for(std::size_t i=0; i<N; ++i)
        ret.array[i] = lhs.array[i] + rhs;
    return ret;
}

template<std::size_t N>
auto operator+(double lhs, const autodiff_tuple<N> &rhs) {
    return rhs + lhs;
}

#endif // AUTODIFF_TUPLE_HPP_INCLUDED
