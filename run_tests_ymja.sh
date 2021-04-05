#!/bin/bash
#python3 src/run_protocol.py IRN_inter_aug configs/YMJA/IRN_inter.cfg YMJA -n 3
#python3 src/run_protocol.py IRN_intra_aug configs/YMJA/IRN_intra.cfg YMJA -n 3
#python3 src/run_protocol.py IRN_inter_no_aug configs/YMJA/IRN_inter-no_aug.cfg YMJA -n 3
#python3 src/run_protocol.py IRN_intra_no_aug configs/YMJA/IRN_intra-no_aug.cfg YMJA -n 3
# python3 src/run_protocol.py IRN_inter_intra configs/YMJA/IRN_inter+intra.cfg YMJA -F middle -n 3
# python3 src/run_protocol.py IRN_fc1_inter_intra configs/YMJA/IRN-fc1_inter+intra.cfg YMJA -F middle -n 3

python3 src/run_protocol.py IRN_joint configs/YMJA/IRN_joint_stream.cfg YMJA -n 3
python3 src/run_protocol.py IRN_temp configs/YMJA/IRN_temporal_stream.cfg YMJA -n 3
python3 src/run_protocol.py IRN_two_stream configs/YMJA/IRN_two_stream.cfg YMJA -F middle -n 3

python3 src/run_protocol.py IRN_joint_no_rel configs/YMJA/IRN_joint_stream_no_rel.cfg YMJA -n 3
python3 src/run_protocol.py IRN_temp_no_rel configs/YMJA/IRN_temporal_stream_no_rel.cfg YMJA -n 3
python3 src/run_protocol.py IRN_two_stream_no_rel configs/YMJA/IRN_two_stream_no_rel.cfg YMJA -F middle -n 3

python3 src/run_protocol.py IRN_joint_att configs/YMJA/IRN_joint_stream_att.cfg YMJA -n 3
python3 src/run_protocol.py IRN_temp_att configs/YMJA/IRN_temporal_stream_att.cfg YMJA -n 3
python3 src/run_protocol.py IRN_two_stream_att configs/YMJA/IRN_two_stream_att.cfg YMJA -F middle -n 3

python3 src/run_protocol.py IRN_joint_att_proj_2000 configs/YMJA/IRN_joint_stream_att_proj_2000.cfg YMJA -n 3
python3 src/run_protocol.py IRN_temp_att_proj_2000 configs/YMJA/IRN_temporal_stream_att_proj_2000.cfg YMJA -n 3
python3 src/run_protocol.py IRN_two_stream_att_proj_2000 configs/YMJA/IRN_two_stream_att_proj_2000.cfg YMJA -F middle -n 3

python3 src/run_protocol.py IRN_joint_att_no_rel configs/YMJA/IRN_joint_stream_att_no_rel.cfg YMJA -n 3
python3 src/run_protocol.py IRN_temp_att_no_rel configs/YMJA/IRN_temporal_stream_att_no_rel.cfg YMJA -n 3
python3 src/run_protocol.py IRN_two_stream_att_no_rel configs/YMJA/IRN_two_stream_att_no_rel.cfg YMJA -F middle -n 3

python3 src/run_protocol.py IRN_joint_att_no_rel_proj_2000 configs/YMJA/IRN_joint_stream_att_no_rel_proj_2000.cfg YMJA -n 3
python3 src/run_protocol.py IRN_temp_att_no_rel_proj_2000 configs/YMJA/IRN_temporal_stream_att_no_rel_proj_2000.cfg YMJA -n 3
python3 src/run_protocol.py IRN_two_stream_att_no_rel_proj_2000 configs/YMJA/IRN_two_stream_att_no_rel_proj_2000.cfg YMJA -F middle -n 3

python3 src/run_protocol.py IRN_two_stream_avg configs/YMJA/IRN_two_stream_avg.cfg YMJA -F middle -n 3
python3 src/run_protocol.py IRN_two_stream_att_avg configs/YMJA/IRN_two_stream_att_avg.cfg YMJA -F middle -n 3
python3 src/run_protocol.py IRN_two_stream_att_no_rel_avg configs/YMJA/IRN_two_stream_att_no_rel_avg.cfg YMJA -F middle -n 3
python3 src/run_protocol.py IRN_two_stream_no_rel_avg configs/YMJA/IRN_two_stream_no_rel_avg.cfg YMJA -F middle -n 3
