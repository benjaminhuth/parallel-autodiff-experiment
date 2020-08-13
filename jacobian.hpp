#ifndef JACOBIAN_HPP_INCLUDED
#define JACOBIAN_HPP_INCLUDED

#include <Eigen/Dense>
#include <xsimd/xsimd.hpp>

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

/// Gradient initialization
template<std::size_t Size>
constexpr auto xsimd_initial_grad(std::size_t pos_of_one)
{
    xsimd::batch<double, Size> batch(0.0);
    batch[pos_of_one] = 1.0;
    return batch;
}

template<Eigen::Index Size>
constexpr auto eigen_initial_grad(std::size_t pos_of_one)
{
    Eigen::Array<double, Size, 1> array(0.0);
    array[pos_of_one] = 1.0;
    return array;
}

/// Enum to choose optimization
enum class optimization
{
    none, eigen, xsimd
};

/// Gradient types for different optimizations
template<optimization opt, Eigen::Index N>
struct gradient_type;

template<Eigen::Index N>
struct gradient_type<optimization::none, N>
{
    using type = double;
};

template<Eigen::Index N>
struct gradient_type<optimization::eigen, N>
{
    using type = Eigen::Array<double, N, 1>;
};

template<Eigen::Index N>
struct gradient_type<optimization::xsimd, N>
{
    using type = xsimd::batch<double, N>;
};

/// Jacobian implementation
template<optimization opt, typename Func_t, Eigen::Index N>
constexpr auto jacobian(const Func_t &f, const Eigen::Matrix<double, N, 1> &input)
{    
    // Ensure that f works as expected
    using Result_t = decltype(f(input));
    
    static_assert(
        Result_t::RowsAtCompileTime != -1 && Result_t::ColsAtCompileTime == 1 &&
        std::is_same_v< Result_t, Eigen::Matrix<double, Result_t::RowsAtCompileTime, 1> >,
        "Result type must be a fixed-sized Eigen::Matrix<double, N, 1> (thus a vector)"
    );
    
    constexpr Eigen::Index JCols = N;
    constexpr Eigen::Index JRows = Result_t::RowsAtCompileTime;
    
    using Dual_t = autodiff::forward::Dual<double, typename gradient_type<opt, JCols>::type>;
    using Jac_t = Eigen::Matrix<double, JRows, JCols, Eigen::RowMajor>;
    
    // Initialize input
    Eigen::Matrix<Dual_t, N, 1> dual_input;
    for(auto i=0ul; i<N; ++i)
        dual_input(i) = input(i);
    
    Jac_t jac;
    
    // No optimization
    if constexpr( opt == optimization::none )
    {
        jac = autodiff::forward::jacobian(f, at(dual_input), wrt(dual_input));
    }
    // Optimized
    else
    {
        for(auto i=0ul; i<JRows; ++i)
            if constexpr( opt == optimization::eigen )
                dual_input(i).grad = eigen_initial_grad<JCols>(i);
            else if constexpr( opt == optimization::xsimd )
                dual_input(i).grad = xsimd_initial_grad<JCols>(i);
    
        auto out = f(dual_input);
        
//         for(auto i=0ul; i<JRows; ++i)
//             std::memcpy(&jac(i,0), &out(i).grad[0], sizeof(double)*JCols);
        
        for(auto i=0ul; i<JRows; ++i)
            for(auto j=0ul; j<JCols; ++j)
                jac(i,j) = out(i).grad[j];
    }
    
    return jac;       
}

#endif // JACOBIAN_HPP_INCLUDED
