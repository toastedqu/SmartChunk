print_date=$(date)
timestamp=$(date -d "$print_date" +"%Y%m%d_%H%M%S")
log_filename="experiment_$timestamp.log"
err_filename="experiment_$timestamp.err"
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "Running experiments at $print_date" 1> $log_filename
# python3 qa_qa.py -c "functools.partial(chunker.cluster_chunker, k=3, mode='k-preserve')" -l 100 1>> $log_filename
# python3 qa_qa.py -c "functools.partial(chunker.cluster_chunker, k=3, mode='k-preserve')" -l 500 1>> $log_filename
# python3 qa_qa.py -c "functools.partial(chunker.arbitrary_chunker, k=100)" -l 100 1>> $log_filename
# python3 qa_qa.py -c "functools.partial(chunker.arbitrary_chunker, k=100)" -l 500 1>> $log_filename
# python3 qa_qa.py -c "functools.partial(chunker.sent_cont_chunker, k=1)" -l 100 1>> $log_filename
# python3 qa_qa.py -c "functools.partial(chunker.sent_cont_chunker, k=1)" -l 500 1>> $log_filename
python3 beir_ir.py -c "functools.partial(chunker.cluster_chunker, k=2, mode='k-preserve')" -d "nfcorpus" -s "test" 1>> $log_filename
python3 beir_ir.py -c "functools.partial(chunker.cluster_chunker, k=3, mode='k-preserve')" -d "nfcorpus" -s "test" 1>> $log_filename
python3 beir_ir.py -c "functools.partial(chunker.cluster_chunker, k=4, mode='k-preserve')" -d "nfcorpus" -s "test" 1>> $log_filename
python3 beir_ir.py -c "functools.partial(chunker.cluster_chunker, k=2, mode='k-split')" -d "nfcorpus" -s "test" 1>> $log_filename
python3 beir_ir.py -c "functools.partial(chunker.cluster_chunker, k=3, mode='k-split')" -d "nfcorpus" -s "test" 1>> $log_filename
python3 beir_ir.py -c "functools.partial(chunker.cluster_chunker, k=4, mode='k-split')" -d "nfcorpus" -s "test" 1>> $log_filename
python3 beir_ir.py -c "functools.partial(chunker.arbitrary_chunker, k=100)" -d "nfcorpus" -s "test" 1>> $log_filename
python3 beir_ir.py -c "functools.partial(chunker.sent_cont_chunker, k=1)" -d "nfcorpus" -s "test" 1>> $log_filename
python3 beir_ir.py -c "chunker.whole_chunker" -d "nfcorpus" -s "test" 1>> $log_filename
