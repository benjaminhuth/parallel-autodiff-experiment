#ifndef AUTODIFF_TUPLE_HPP_INCLUDED
#define AUTODIFF_TUPLE_HPP_INCLUDED

#include <array>

template<std::size_t N>
struct grad_array
{
    std::array<double, N> array;
    
    grad_array() = default;
    
    grad_array &operator=(const grad_array &) = default; 	
    
    grad_array(double a) {
        for(std::size_t i=0; i<N; ++i)
            array[i] = a;
    }
    
    template<std::size_t M>
    void set_to_one_all_others_zero() {   
        static_assert(M < N, "access error");
        for(std::size_t i=0; i<N; ++i)
            if( i == M )
                array[i] = 1.0;
            else
                array[i] = 0.0;
    }
    
    // OPERATOR *=
    grad_array& operator*=(const grad_array& rhs)
    {
        for(std::size_t i=0; i<N; ++i)
            array[i] = array[i] * rhs.array[i];
        
        return *this;
    }
    
    grad_array& operator*=(double rhs)
    {
        for(std::size_t i=0; i<N; ++i)
            array[i] = array[i] * rhs;
        
        return *this;
    }
    
    // OPERATOR +=
    grad_array& operator+=(const grad_array& rhs)
    {
        for(std::size_t i=0; i<N; ++i)
            array[i] = array[i] + rhs.array[i];
        
        return *this;
    }
    
    grad_array& operator+=(double rhs)
    {
        for(std::size_t i=0; i<N; ++i)
            array[i] = array[i] + rhs;
        
        return *this;
    }
    
    
    
};

// OPERATOR *
template<std::size_t N>
auto operator*(const grad_array<N> &lhs, const grad_array<N> &rhs) {
    grad_array<N> ret;
    for(std::size_t i=0; i<N; ++i)
        ret.array[i] = lhs.array[i] * rhs.array[i];
    return ret;
}

template<std::size_t N>
auto operator*(const grad_array<N> &lhs, double rhs) {
    grad_array<N> ret;
    for(std::size_t i=0; i<N; ++i)
        ret.array[i] = lhs.array[i] * rhs;
    return ret;
}

template<std::size_t N>
auto operator*(double lhs, const grad_array<N> &rhs) {
    return rhs * lhs;
}

// OPERATOR +
template<std::size_t N>
auto operator+(const grad_array<N> &lhs, const grad_array<N> &rhs) {
    grad_array<N> ret;
    for(std::size_t i=0; i<N; ++i)
        ret.array[i] = lhs.array[i] + rhs.array[i];
    return ret;
}

template<std::size_t N>
auto operator+(const grad_array<N> &lhs, double rhs) {
    grad_array<N> ret;
    for(std::size_t i=0; i<N; ++i)
        ret.array[i] = lhs.array[i] + rhs;
    return ret;
}

template<std::size_t N>
auto operator+(double lhs, const grad_array<N> &rhs) {
    return rhs + lhs;
}

#endif // AUTODIFF_TUPLE_HPP_INCLUDED
