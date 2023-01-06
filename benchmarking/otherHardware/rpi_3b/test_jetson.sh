echo 'Starting Test for Raspberry Pi 3 Model B'
echo ''
echo 'Nonspiking Dense:'
python3 test_rpi3b_nonspiking_dense.py
echo ''
echo 'Nonspiking Sparse:'
python3 test_rpi3b_nonspiking_sparse.py
echo ''
echo 'Spiking Dense:'
python3 test_rpi3b_spiking_dense.py
echo ''
echo 'Spiking Sparse:'
python3 test_rpi3b_spiking_sparse.py
echo ''
echo 'Testing finished, all data recorded'