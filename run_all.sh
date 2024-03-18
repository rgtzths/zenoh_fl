s=0.95
p=10
md=0.01
datasets=("IOT_DNL" "Slicing5G")
learning_rates=(0.000005 0.00001)
n_workers=(8 4 2)

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    lr=${learning_rates[$i]}
    for n in "${n_workers[@]}"; do
        n=$((n+1))

        # centralized
        e=100
        batch_sizes=(128 256 512 1024 2048 4096 8192)
        for b in "${batch_sizes[@]}"; do
            # async
            command="mpirun -np $n -hostfile FL/host_file python3 federated_learning.py -m 1 \
            -d $dataset -lr $lr -b $b -s $s -p $p -md $md -e $e"
            eval $command
            # sync
            command="mpirun -np $n -hostfile FL/host_file python3 federated_learning.py -m 2 \
            -d $dataset -lr $lr -b $b -s $s -p $p -md $md -e $e"
            eval $command
        done

        # decentralized
        batch_sizes=(512, 1024, 2048, 4096)
        global_epochs=(300 100 50 25 10)
        local_epochs=(1 3 6 12 30)
        for b in "${batch_sizes[@]}"; do
            for i in "${!global_epochs[@]}"; do
                ge=${global_epochs[$i]}
                le=${local_epochs[$i]}
                # async
                alphas=(0.1 0.2 0.3 0.4 0.5)
                for a in "${alphas[@]}"; do
                    command="mpirun -np $n -hostfile FL/host_file python3 federated_learning.py -m 3 \
                    -d $dataset -lr $lr -b $b -s $s -p $p -md $md -ge $ge -le $le -a $a"
                    eval $command
                done
                # sync
                command="mpirun -np $n -hostfile FL/host_file python3 federated_learning.py -m 4 \
                -d $dataset -lr $lr -b $b -s $s -p $p -md $md -ge $ge -le $le"
                eval $command
            done
        done
    done
done 