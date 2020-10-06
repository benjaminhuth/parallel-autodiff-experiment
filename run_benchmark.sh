#!/bin/sh

if [[ -z "$1" || -z "$2" ]]; then
    echo "Usage $0 <compiler+optflags> <iterations> <options>"
    exit 0
fi

echo "size,iter,normal_us,eigen_us,eigen_speedup,xsimd_us,xsimd_speedup"

for i in {1..32}; do
    $1 -I dependencies/autodiff -std=c++17 main.cpp -D JACOBIAN_SIZE=$i
    RES=$(./a.out $2 --csv $3)
    echo "$i, $2, $RES"
done
