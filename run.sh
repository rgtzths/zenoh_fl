#!/bin/bash
source venv/bin/activate

while getopts m:d:g:l:r:b:o:s:p:a:e: flag
do
    case "${flag}" in
        m) m="${OPTARG}";;
        d) d="${OPTARG}";;
        g) g="${OPTARG}";;
        l) l="${OPTARG}";;
        r) r="${OPTARG}";;
        b) b="${OPTARG}";;
        o) o="${OPTARG}";;
        s) s="${OPTARG}";;
        p) p="${OPTARG}";;
        a) a="${OPTARG}";;
        e) e="${OPTARG}";;
        *) echo "Invalid option: -${OPTARG}" >&2; exit 1;;
    esac
done

#echo "python3 federated_learning.py -m $m -d $d -ge $g -le $l -lr $r -b $b -o $o -s $s -p $p"
python3 federated_learning.py -m $m -d $d -ge $g -le $l -lr $r -b $b -o $o -s $s -p $p -a $a --seed $e;