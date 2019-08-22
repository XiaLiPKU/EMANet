#! /bin/bash
function rand() {
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000000))
    echo $(($num%$max+$min))
}

rnd=$(rand 6000 12000)
tensorboard --logdir ../logdir --host 0.0.0.0 --port $rnd
