echo 'Starting Test for Jetson Nano'
echo ''
echo 'Nonspiking Dense:'
python3 test_jetson_nonspiking_dense_large.py
echo ''
echo 'Nonspiking Sparse:'
python3 test_jetson_nonspiking_sparse_large.py
echo ''
echo 'Testing finished, all data recorded'