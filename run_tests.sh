#!/bin/bash
python3 src/run_protocol.py IRN_inter_aug configs/SBU/IRN_inter.cfg SBU -n 1
python3 src/run_protocol.py IRN_intra_aug configs/SBU/IRN_intra.cfg SBU -n 1
python3 src/run_protocol.py IRN_inter_no_aug configs/SBU/IRN_inter-no_aug.cfg SBU -n 1
python3 src/run_protocol.py IRN_intra_no_aug configs/SBU/IRN_intra-no_aug.cfg SBU -n 1
python3 src/run_protocol.py IRN_inter_intra configs/SBU/IRN_inter+intra.cfg SBU -F middle -n 1
python3 src/run_protocol.py IRN_fc1_inter_intra configs/SBU/IRN-fc1_inter+intra.cfg SBU -F middle -n 1
python3 src/run_protocol.py IRN_naive_inter_intra configs/SBU/Naive-IRN_inter+intra.cfg SBU -n 1
python3 src/run_protocol.py IRN_inter_random configs/SBU/IRN_inter_random.cfg SBU -n 1
python3 src/run_protocol.py IRN_intra_random configs/SBU/IRN_intra_random.cfg SBU -n 1
python3 src/run_protocol.py IRN_inter_intra_random configs/SBU/IRN_inter+intra_random.cfg SBU -F middle -n 1

python3 src/run_protocol.py IRN_two_stream configs/SBU/IRN_two_stream.cfg SBU -F middle -n 1
