#!/bin/bash
# python3 src/run_protocol.py IRN_inter_aug configs/SBU/IRN_inter.cfg SBU -n 1
# python3 src/run_protocol.py IRN_intra_aug configs/SBU/IRN_intra.cfg SBU -n 1
# python3 src/run_protocol.py IRN_inter_no_aug configs/SBU/IRN_inter-no_aug.cfg SBU -n 1
# python3 src/run_protocol.py IRN_intra_no_aug configs/SBU/IRN_intra-no_aug.cfg SBU -n 1
# python3 src/run_protocol.py IRN_inter_intra configs/SBU/IRN_inter+intra.cfg SBU -F middle -n 1
# python3 src/run_protocol.py IRN_fc1_inter_intra configs/SBU/IRN-fc1_inter+intra.cfg SBU -F middle -n 1
# python3 src/run_protocol.py IRN_naive_inter_intra configs/SBU/Naive-IRN_inter+intra.cfg SBU -n 1
# python3 src/run_protocol.py IRN_inter_random configs/SBU/IRN_inter_random.cfg SBU -n 1
# python3 src/run_protocol.py IRN_intra_random configs/SBU/IRN_intra_random.cfg SBU -n 1
# python3 src/run_protocol.py IRN_inter_intra_random configs/SBU/IRN_inter+intra_random.cfg SBU -F middle -n 1

# python3 src/run_protocol.py IRN_te configs/SBU/IRN_temporal_stream.cfg SBU -n 1
# python3 src/run_protocol.py IRN_two_stream configs/SBU/IRN_two_stream.cfg SBU -F middle -n 1

python3 src/run_protocol.py IRN_joint_att configs/SBU/IRN_joint_stream_att.cfg SBU -n 1
python3 src/run_protocol.py IRN_temp_att configs/SBU/IRN_temporal_stream_att.cfg SBU -n 1
python3 src/run_protocol.py IRN_two_stream_att configs/SBU/IRN_two_stream_att.cfg SBU -F middle -n 1

python3 src/run_protocol.py IRN_joint_att_proj_2000 configs/SBU/IRN_joint_stream_att_proj_2000.cfg SBU -n 1
python3 src/run_protocol.py IRN_temp_att_proj_2000 configs/SBU/IRN_temporal_stream_att_proj_2000.cfg SBU -n 1
python3 src/run_protocol.py IRN_two_stream_att_proj_2000 configs/SBU/IRN_two_stream_att_proj_2000.cfg SBU -F middle -n 1

python3 src/run_protocol.py IRN_joint_att_no_rel configs/SBU/IRN_joint_stream_att_no_rel.cfg SBU -n 1
python3 src/run_protocol.py IRN_temp_att_no_rel configs/SBU/IRN_temporal_stream_att_no_rel.cfg SBU -n 1
python3 src/run_protocol.py IRN_two_stream_att_no_rel configs/SBU/IRN_two_stream_att_no_rel.cfg SBU -F middle -n 1

python3 src/run_protocol.py IRN_joint_att_no_rel_proj_2000 configs/SBU/IRN_joint_stream_att_no_rel_proj_2000.cfg SBU -n 1
python3 src/run_protocol.py IRN_temp_att_no_rel_proj_2000 configs/SBU/IRN_temporal_stream_att_no_rel_proj_2000.cfg SBU -n 1
python3 src/run_protocol.py IRN_two_stream_att_no_rel_proj_2000 configs/SBU/IRN_two_stream_att_no_rel_proj_2000.cfg SBU -F middle -n 1


# python3 src/run_protocol.py IRN_two_stream_avg configs/SBU/IRN_two_stream_avg.cfg SBU -F middle -n 1
python3 src/run_protocol.py IRN_two_stream_att_avg configs/SBU/IRN_two_stream_att_avg.cfg SBU -F middle -n 1
python3 src/run_protocol.py IRN_two_stream_att_no_rel_avg configs/SBU/IRN_two_stream_att_no_rel_avg.cfg SBU -F middle -n 1
# python3 src/run_protocol.py IRN_two_stream_no_rel_avg configs/SBU/IRN_two_stream_no_rel_avg.cfg SBU -F middle -n 1