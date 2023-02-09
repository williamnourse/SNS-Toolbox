echo 'Starting Test for Brian2CUDA'
echo ''
echo 'Nonspiking Dense:'
python3 test_brian2cuda_nonspiking_dense.py
echo ''
echo 'Nonspiking Sparse:'
python3 test_brian2cuda_nonspiking_sparse.py
echo ''
echo 'Spiking Sparse:'
python3 test_brian2cuda_spiking_sparse.py
echo ''
echo 'Spiking Dense:'
python3 test_brian2cuda_spiking_dense.py
echo ''
echo 'Testing finished, all data recorded'