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


#ifndef JACOBIAN_SIZE
#define JACOBIAN_SIZE 8
#endif
constexpr int N = JACOBIAN_SIZE;

template<typename JacFunc, typename GenFunc, typename Jacobian>
auto benchmark(JacFunc &f, GenFunc &g, Jacobian &j, int iterations, bool use_data_vector)
{
    using vec_t = Eigen::Matrix<double,N,1>;
    using vecAD_t = decltype(g(vec_t()));
    
    std::chrono::high_resolution_clock::time_point t0, t1;
    
    if( use_data_vector )
    {
        std::vector<vec_t> data(iterations);
        std::vector<vecAD_t> xs(iterations);
        std::generate(data.begin(),data.end(),[](){ return vec_t::Random(); }); 
        std::transform(data.begin(),data.end(),xs.begin(),[&](const auto &a){ return g(a); });
        
        t0 = std::chrono::high_resolution_clock::now();
        
        for(std::size_t i=0; i<iterations; ++i)
            j=f(xs[i]);
        
        t1 = std::chrono::high_resolution_clock::now();
    }
    else
    {
        vec_t vec;
        for(std::size_t i=0; i<vec.size(); ++i)
            vec(i) = std::log(iterations)*i;
        auto x = g(vec);
        
        t0 = std::chrono::high_resolution_clock::now();
        
        for(std::size_t i=0; i<iterations; ++i)
        {
            j=f(x);
            asm volatile("" ::: "memory");
        }
        
        t1 = std::chrono::high_resolution_clock::now();
    }
    
    return std::chrono::duration<double>(t1 - t0);
}

int main(int argc, char ** argv) 
{    
    
    std::vector<std::string> args(argv, argv+argc);
    
    // help
    if( std::any_of(args.begin(), args.end(), [](const auto &arg){ return arg == "-h" || arg == "--help"; }) )
    {
        std::cout << "Usage: " << argv[0] << " <iterations> <options>\n\n";
        std::cout << "Options:\n";
        std::cout << "\t--csv           csv-readable output\n";
        std::cout << "\t--multi-data    use individual input for each iteration\n";
        std::cout << std::endl;
        return 0;
    }
    
    // iterations
    int iterations = 100;
    if( argc >= 2 )
        iterations = std::atoi(argv[1]);
    
    // the function
    auto f = [](const auto &x){ return typename std::remove_reference<decltype(x)>::type{ x.dot(x) * x }; };
    
    // init x vector
    Eigen::Matrix<double, N, 1> in;
    double val = argc+1;
    for(std::size_t i=0; i<in.size(); ++i)
        in(i) = val*val + i*val;
    
    // normal routine
    auto gen_normal = [](const auto &in)
    {
        Eigen::Matrix<autodiff::dual, N, 1> x;
        for(std::size_t i=0; i<in.size(); ++i)
            x(i) = in(i);
        
        return x;
    };
    
    auto comp_normal = [&f](auto &x)
    {
        using namespace autodiff::forward;
        return jacobian(f, wrt(x), at(x));        
    };
    
    // new routine
    auto gen_new = [](const auto &in)
    {
        using Dual = autodiff::forward::Dual<double, Eigen::Array<double, N, 1>>;
        
        Eigen::Matrix<Dual, N, 1> x;
        for(std::size_t i=0; i<in.size(); ++i)
            x(i) = in(i);
        
        return x;
    };
    auto comp_new = [&f](auto &x)
    {        
        return jacobian(f, x);        
    };
    
    // Do actual benchmark
    Eigen::Matrix<double, N, N> jac_normal, jac_new;
    bool use_multi_data = std::any_of(args.begin(), args.end(), [](const auto &arg){ return arg == "--multi-data"; });
    
    double normal_us = benchmark(comp_normal, gen_normal, jac_normal, iterations, use_multi_data).count()*1.e6;
    double new_us = benchmark(comp_new, gen_new, jac_new, iterations, use_multi_data).count()*1.e6;
    
    if( std::any_of(args.begin(), args.end(), [](const auto &arg){ return arg == "--csv"; }) )
    {
        std::cout << normal_us << ", " << new_us << ", " << normal_us/new_us << std::endl;
    }
    else
    {    
        std::cout << "jacobian size = " << N << "x" << N << std::endl;
        std::cout << "normal method: " << normal_us << " us" << std::endl;
        std::cout << "NEW method:    " << new_us << " us" << std::endl;
        std::cout << "speedup:      x" << normal_us/new_us << std::endl;
    }
    
    if( !use_multi_data && (jac_normal - jac_new).norm() > 1.e-3 )
        throw std::runtime_error("validation error!");
}
