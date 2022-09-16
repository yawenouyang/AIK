for ((split_index=0; split_index<5; ++split_index)); do
    for ((seed=0; seed<5; ++seed)); do
        python train_aik.py --gpu 0 --seed $seed --split_index $split_index --dataset mixsnips
        python train_aik.py --gpu 0 --seed $seed --split_index $split_index --dataset multiwoz23
    done
done