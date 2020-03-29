# https://dscomp.nkn.uidaho.edu/data

My sad sad entry into the UIdaho Data Science Competition for 2020

But hey 20 points of extra credit

## DLAMI Errata

    source activate tensorflow2_p36

Make sure you're using tensorflow2! If you see a bunch of warnings about various methods it's likely you sourced the wrong environment.

    git clone https://github.com/idahood/dscomp2020.git
    ./init.sh
    ./my_model --batch 128 --epoch 128 --split 0.15
    ./grading.py models/model-$ID

## Improvement Ideas

Augment training data with lateral shift deformation?
Add class for empty?
