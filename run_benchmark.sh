#!/bin/sh

if [[ -z "$1" || -z "$2" ]]; then
    echo "Usage $0 <compiler> <iterations>"
    exit 0
fi

echo "Jac_size,iter,normal_us,new_us,speedup"

for i in {1..16}; do
    $1 -O3 -std=c++17 main.cpp -D JACOBIAN_SIZE=$i
    RES=$(./a.out $2 --csv)
    echo "$i, $2, $RES"
done
