s=0.96
p=300
md=0.0001
datasets=("Slicing5G" "IOT_DNL")
learning_rates=(0.000005 0.00001)
n_workers=(2)

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    lr=${learning_rates[$i]}
    for n in "${n_workers[@]}"; do
        n=$((n+1))

        # centralized
        e=300
        batch_sizes=(32 64 128 256 512 1024 2048)
        for b in "${batch_sizes[@]}"; do
            # async
            command="mpirun -np $n python federated_learning.py -m 1 \
            -d $dataset -lr $lr -b $b -s $s -p $p -md $md -e $e"
            eval $command
            # sync
            command="mpirun -np $n python federated_learning.py -m 2 \
            -d $dataset -lr $lr -b $b -s $s -p $p -md $md -e $e"
            eval $command
        done

        # decentralized
        batch_sizes=(32 64 128 256 512 1024)
        ge=300
        local_epochs=(1 2 4 8 16)
        for b in "${batch_sizes[@]}"; do
            for le in "${local_epochs[@]}"; do
                # async
                alphas=(0.1 0.2 0.3 0.4 0.5)
                for a in "${alphas[@]}"; do
                    command="mpirun -np $n python federated_learning.py -m 3 \
                    -d $dataset -lr $lr -b $b -s $s -p $p -md $md -ge $ge -le $le -a $a"
                    eval $command
                done
                # sync
                command="mpirun -np $n python federated_learning.py -m 4 \
                -d $dataset -lr $lr -b $b -s $s -p $p -md $md -ge $ge -le $le"
                eval $command
            done
        done
    done
done 