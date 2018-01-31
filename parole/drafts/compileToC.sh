#/bin/bash

pythonver=`ls /usr/include/ | grep python3`

echo $pythonver

cython perceptron_parole.py -o perceptron_parole.c  --embed

gcc -Os -I /usr/include/$pythonver perceptron_parole.c -o perceptron_parole -l$pythonver -lpthread -lm -lutil -ldl

rm perceptron_parole.c
