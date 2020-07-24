#include <iostream>
#include <Eigen/Core>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#ifndef JACOBIAN_SIZE
#define JACOBIAN_SIZE 8
#endif

#include <chrono>

/******************************************************
 * 
 * 
 *    f( x, y ) = { f_1, f_2 }
 * 
 *    new dual, where grad = std::array<double, N>
 * 
 * 
 *    f(  (x,[1,0]) , (y,[0,1])  ) 
 * 
 *    = 
 * 
 *    {   
 *        f_1(x,y),   [ df_1/dx, df_1/dy ],
 *        f_2(x,y),   [ df_2/dx, df_2/dy ]
 *    }
 * 
 * 
 ******************************************************/

template<int I, int I_end, typename Body>
void constexpr_for(const Body &loop_body)
{
    static_assert(std::is_same<decltype(loop_body(0)), void>::value, "Loop body function not valid");
    loop_body(I);
    if constexpr( I < I_end-1 )
        constexpr_for<I+1, I_end>(loop_body);
}

template<int Size>
constexpr auto initial_grad(std::size_t pos_of_one)
{
    Eigen::Array<double, Size, 1> array(0.0);
    array[pos_of_one] = 1.0;
    return array;
}

template<typename Func_t, typename Input_t>
constexpr auto jacobian(Func_t &f, Input_t &in)
{
    using Result_t = decltype(f(in));
    using Gradient_t = decltype(Input_t::value_type::grad);
    
    static_assert( 
        std::is_same<Gradient_t, typename Gradient_t::PlainArray>::value,
        "Autodiff gradient type must be an Eigen::Array type"
    );
    static_assert(
        Gradient_t::ColsAtCompileTime == 1 && 
        static_cast<int>(Gradient_t::RowsAtCompileTime) == static_cast<int>(Input_t::RowsAtCompileTime),
        "Autodiff gradient type must be a fixed-size 1d array with the same size as the output type"
    );
    static_assert(
        Input_t::ColsAtCompileTime == 1 && Input_t::RowsAtCompileTime != Eigen::Dynamic,
        "Input type must be a fixed-size column vector"
    );
    static_assert(
        Result_t::ColsAtCompileTime == 1 && Result_t::RowsAtCompileTime != Eigen::Dynamic,
        "Output type must be a fixed-size column vector"
    );
    
    using Jac_t = Eigen::Matrix<double, Result_t::RowsAtCompileTime, Input_t::RowsAtCompileTime>;
    
    constexpr_for<0, Input_t::RowsAtCompileTime>([&](auto i){
        in(i).grad = initial_grad<Input_t::RowsAtCompileTime>(i);
    });
    
    auto out = f(in);
    Jac_t jac;
        
    constexpr_for<0, Input_t::RowsAtCompileTime>([&](auto i){
        jac.row(i) = out(i).grad;
    });
    
    return jac;       
}

template<typename JacobianComputeFunc, typename Jacobian>
auto benchmark(JacobianComputeFunc &f, Jacobian &j, bool dummy_bool, int iterations)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for(std::size_t i=0; i<iterations; ++i)
    {
        j=f();
        // asm no-op to prevent optimization
        __asm__ __volatile__ ("" : : : "memory");
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double>(t1 - t0)/* / iterations*/;
}

#ifndef JACOBIAN_SIZE
#define JACOBIAN_SIZE 8
#endif

int main(int argc, char ** argv) 
{    
    constexpr int N = JACOBIAN_SIZE;
    
    // the function
    auto f = [](const auto &x){ return typename std::remove_reference<decltype(x)>::type{ x.dot(x) * x }; };
    
    // init x vector
    Eigen::Matrix<double, N, 1> in;
    double val = argc+1;
    for(std::size_t i=0; i<in.size(); ++i)
        in(i) = val*val + i*val;
    
    // normal routine
    auto jacobian_normal = [&]()
    {
        Eigen::Matrix<autodiff::dual, N, 1> x;
        for(std::size_t i=0; i<in.size(); ++i)
            x(i) = in(i);
        
        return autodiff::forward::jacobian(f, wrt(x), at(x));        
    };
    
    // new routine
    auto jacobian_new = [&]()
    {
        using Dual = autodiff::forward::Dual<double, Eigen::Array<double, N, 1>>;
        
        Eigen::Matrix<Dual, N, 1> x;
        for(std::size_t i=0; i<in.size(); ++i)
            x(i) = in(i);
        
        return jacobian(f, x);        
    };
    
    Eigen::Matrix<double, N, N> jac_normal, jac_new;
    
    // benchmark
    bool dummy_bool = argc>100;
    int iterations = 100;
    if( argc >= 2 )
        iterations = std::atoi(argv[1]);
    
    double normal_us = benchmark(jacobian_normal, jac_normal, dummy_bool, iterations).count()*1.e6;
    double new_us = benchmark(jacobian_new, jac_new, dummy_bool, iterations).count()*1.e6;
    
    if( argc == 3 && std::string(argv[2]) == "--csv" )
        std::cout << normal_us << ", " << new_us << ", " << normal_us/new_us << std::endl;
    else
    {    
        std::cout << "jacobian size = " << N << "x" << N << std::endl;
        std::cout << "normal method: " << normal_us << " us" << std::endl;
        std::cout << "NEW method:    " << new_us << " us" << std::endl;
        std::cout << "speedup:      x" << normal_us/new_us << std::endl;
    }
    
    if( (jac_normal - jac_new).norm() > 1.e-3 )
        throw std::runtime_error("validation error!");
}
