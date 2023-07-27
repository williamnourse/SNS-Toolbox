echo 'Starting Test for Jetson Nano'
echo ''
echo 'Nonspiking Dense:'
python3 test_jetson_nonspiking_dense.py
echo ''
echo 'Nonspiking Sparse:'
python3 test_jetson_nonspiking_sparse.py
echo ''
echo 'Spiking Dense:'
python3 test_jetson_spiking_dense.py
echo ''
echo 'Spiking Sparse:'
python3 test_jetson_spiking_sparse.py
echo ''
echo 'Testing finished, all data recorded'