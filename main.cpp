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

#include <iostream>
#include <cstdlib>
#include <chrono>

#include "jacobian.hpp"

#ifndef JACOBIAN_SIZE
#define JACOBIAN_SIZE 4
#endif

// Determine if XSIMD can be used
#if ( defined(__SSE__)     && JACOBIAN_SIZE == 2 ) || \
    ( defined(__AVX2__)    && JACOBIAN_SIZE == 4 ) || \
    ( defined(__AVX512F__) && JACOBIAN_SIZE == 8 )
#define USE_XSIMD
#endif
   
// XSIMD can be disabled by NO_XSIMD
#if defined(USE_XSIMD) && defined(NO_XSIMD)
#undef USE_XSIMD
#endif

constexpr int N = JACOBIAN_SIZE;

using Jac_t = Eigen::Matrix<double, N, N, Eigen::RowMajor>;
using Vec_t = Eigen::Matrix<double, N, 1>;


// Benchmark without single data
template<optimization opt, typename Func_t>
auto benchmark(const Func_t &f, Jac_t &j, std::size_t iterations)
{
    std::srand(iterations);
    
    const auto x = Vec_t::Random();
        
    auto t0 = std::chrono::high_resolution_clock::now();
        
    for(auto i=0ul; i<iterations; ++i)
    {
        j = jacobian<opt, Func_t, N>(f, x);
        asm volatile("" ::: "memory");
    }
        
    auto t1 = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double>(t1 - t0);
}

// Benchmark with multiple data
template<optimization opt, typename Func_t>
auto benchmark(const Func_t &f, std::vector<Jac_t> &js, std::size_t iterations)
{
    std::srand(iterations);
    
    std::vector<Vec_t> x(iterations);
    std::generate(x.begin(),x.end(),[](){ return Vec_t::Random(); }); 
        
    auto t0 = std::chrono::high_resolution_clock::now();
        
    for(auto i=0ul; i<iterations; ++i)
    {
        js[i] = jacobian<opt, Func_t, N>(f, x[i]);
        asm volatile("" ::: "memory");
    }
        
    auto t1 = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double>(t1 - t0);
}


int main(int argc, char ** argv) 
{
    std::vector<std::string> args(argv, argv+argc);
    
    // Help
    if( std::any_of(args.begin(), args.end(), [](const auto &arg){ return arg == "-h" || arg == "--help"; }) )
    {
        std::cout << "Usage: " << argv[0] << " <iterations> <options>\n\n";
        std::cout << "Options:\n";
        std::cout << "\t--csv           csv-readable output\n";
        std::cout << "\t--multi-data    use individual input for each iteration\n";
        std::cout << std::endl;
        return 0;
    }
    
    // Iterations
    std::size_t iterations = 1000;
    if( argc >= 2 )
        iterations = std::atoi(argv[1]);
    
    const Jac_t rand_mat1 = Jac_t::Random();
    const Jac_t rand_mat2 = Jac_t::Random();
    
    // The function
    auto f = [&](const auto &x)
    { 
        using namespace std;
        using namespace xsimd;
        return typename std::remove_reference<decltype(x)>::type( (rand_mat1*x).array() * (rand_mat2*x).array() ); 
    };
    
    // Do actual benchmark
    Jac_t jac_normal, jac_eigen, jac_xsimd;
    std::vector<Jac_t> jacs_normal(iterations), jacs_eigen(iterations), jacs_xsimd(iterations);
    
    const bool use_multi_data = std::any_of(args.begin(), args.end(), [](const auto &arg){ return arg == "--multi-data"; });
    
    double normal_us = 0.0, eigen_us = 0.0, xsimd_us = 0.0;
    
    if( use_multi_data )
    {
#ifndef NO_NORMAL
        normal_us = benchmark<optimization::none>(f, jacs_normal, iterations).count()*1.e6;
#endif
#ifndef NO_EIGEN
        eigen_us = benchmark<optimization::eigen>(f, jacs_eigen, iterations).count()*1.e6;
#endif
#ifdef USE_XSIMD
        xsimd_us = benchmark<optimization::xsimd>(f, jacs_xsimd, iterations).count()*1.e6;
#endif
        
        jac_normal = jacs_normal[0];
        jac_eigen = jacs_eigen[0];
        jac_xsimd = jacs_xsimd[0];
    }
    else
    {
#ifndef NO_NORMAL
        normal_us = benchmark<optimization::none>(f, jac_normal, iterations).count()*1.e6;
#endif
#ifndef NO_EIGEN
        eigen_us = benchmark<optimization::eigen>(f, jac_eigen, iterations).count()*1.e6;
#endif
#ifdef USE_XSIMD
        xsimd_us = benchmark<optimization::xsimd>(f, jac_xsimd, iterations).count()*1.e6;
#endif
    }
    
    if( std::any_of(args.begin(), args.end(), [](const auto &arg){ return arg == "--csv"; }) )
    {
        std::cout << normal_us << "," 
                  << eigen_us << "," << normal_us/eigen_us << "," 
                  << xsimd_us << "," << normal_us/xsimd_us << std::endl;
    }
    else
    {
        std::cout << "INFO: max xsimd::batch<double> size: " << xsimd::simd_traits<double>::size << std::endl;   
        std::cout << "INFO: jacobian size = " << N << "x" << N << std::endl;
#ifndef NO_NORMAL
        std::cout << "no optimization:    " << normal_us << " us" << std::endl;
#endif
#ifndef NO_EIGEN
        std::cout << "eigen optimization: " << eigen_us << " us   (speedup: " << normal_us/eigen_us << ")" << std::endl;
#endif
#ifdef USE_XSIMD
        std::cout << "xsimd optimization: " << xsimd_us << " us   (speedup: " << normal_us/xsimd_us << ")" << std::endl;
#endif
    }
    
#if !defined(NO_NORMAL) && !defined(NO_EIGEN)
    if ( 
        (jac_normal - jac_eigen).norm() > 1.e-3 
#ifdef USE_XSIMD
        || (jac_normal - jac_xsimd ).norm() > 1.e-3 
#endif
    )
    {
        std::cout << "J_normal = \n" << jac_normal << std::endl;
        std::cout << "J_eigen = \n" << jac_eigen << std::endl;
#ifdef USE_XSIMD
        std::cout << "J_xsimd = \n" << jac_xsimd << std::endl;
#endif
        throw std::runtime_error("validation error!");
    }
#endif
}
