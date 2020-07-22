#include <iostream>
#include <Eigen/Core>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#include "autodiff_tuple.hpp"

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



//                                    1-dim value      2-dim grad, this type has all necessary overloads
//                                         |            |
//                                         V            V
using my_dual2 = autodiff::forward::Dual<double, autodiff_tuple<2>>;

using MyVector2 = Eigen::Matrix<my_dual2, 2, 1>;

// This function could be made generic with template arguments and constexpr for loop
template<typename Func>
auto my_jacobian(Func &f, MyVector2 &in)
{
    // set elements in grad to 0 bzw. 1
    in(0).grad.set_to_one<0>();
    in(1).grad.set_to_one<1>();
    
    // compute function
    MyVector2 F = f(in);
    
    Eigen::Matrix2d jac;
    
    // Read out values
    jac(0,0) = F(0).grad.array[0];
    jac(0,1) = F(0).grad.array[1];
    jac(1,0) = F(1).grad.array[0];
    jac(1,1) = F(1).grad.array[1];
    
    return jac;
}

int main(int argc, char **argv) 
{
    auto f = [](MyVector2 &x) { return MyVector2{ x[0]*x[1], 2*x[0]*x[1] }; };
    
    MyVector2 in;
    in(0) = 5.0;
    in(1) = 2.0;
    
    std::cout << "j = \n" << my_jacobian(f, in) << std::endl;
}
