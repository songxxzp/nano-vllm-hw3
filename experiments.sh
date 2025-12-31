# TASK 3.1

python experiments.py --test mmlu
python experiments.py --test mmlu --quant tensor --dtype int8
python experiments.py --test mmlu --quant row --dtype int8
python experiments.py --test mmlu --quant group --group-size 64 --dtype int8
python experiments.py --test mmlu --quant group --group-size 128 --dtype int8
python experiments.py --test mmlu --quant group --group-size 256 --dtype int8
python experiments.py --test mmlu --quant group --group-size 512 --dtype int8
python experiments.py --test mmlu --quant tensor --dtype fp8
python experiments.py --test mmlu --quant row --dtype fp8
python experiments.py --test mmlu --quant group --group-size 64 --dtype fp8
python experiments.py --test mmlu --quant group --group-size 128 --dtype fp8
python experiments.py --test mmlu --quant group --group-size 256 --dtype fp8
python experiments.py --test mmlu --quant group --group-size 512 --dtype fp8

python experiments.py --test mmlu --quant tensor --dtype int8 --real
python experiments.py --test mmlu --quant row --dtype int8 --real
python experiments.py --test mmlu --quant group --group-size 64 --dtype int8 --real
python experiments.py --test mmlu --quant group --group-size 128 --dtype int8 --real
python experiments.py --test mmlu --quant group --group-size 256 --dtype int8 --real
python experiments.py --test mmlu --quant group --group-size 512 --dtype int8 --real
python experiments.py --test mmlu --quant tensor --dtype fp8 --real
python experiments.py --test mmlu --quant row --dtype fp8 --real
python experiments.py --test mmlu --quant group --group-size 64 --dtype fp8 --real
python experiments.py --test mmlu --quant group --group-size 128 --dtype fp8 --real
python experiments.py --test mmlu --quant group --group-size 256 --dtype fp8 --real
python experiments.py --test mmlu --quant group --group-size 512 --dtype fp8 --real

python experiments.py --test ppl
python experiments.py --test ppl --quant tensor --dtype int8
python experiments.py --test ppl --quant row --dtype int8
python experiments.py --test ppl --quant group --group-size 64 --dtype int8
python experiments.py --test ppl --quant group --group-size 128 --dtype int8
python experiments.py --test ppl --quant group --group-size 256 --dtype int8
python experiments.py --test ppl --quant group --group-size 512 --dtype int8
python experiments.py --test ppl --quant tensor --dtype fp8
python experiments.py --test ppl --quant row --dtype fp8
python experiments.py --test ppl --quant group --group-size 64 --dtype fp8
python experiments.py --test ppl --quant group --group-size 128 --dtype fp8
python experiments.py --test ppl --quant group --group-size 256 --dtype fp8
python experiments.py --test ppl --quant group --group-size 512 --dtype fp8

python experiments.py --test ppl --quant tensor --dtype int8 --real
python experiments.py --test ppl --quant row --dtype int8 --real
python experiments.py --test ppl --quant group --group-size 64 --dtype int8 --real
python experiments.py --test ppl --quant group --group-size 128 --dtype int8 --real
python experiments.py --test ppl --quant group --group-size 256 --dtype int8 --real
python experiments.py --test ppl --quant group --group-size 512 --dtype int8 --real
python experiments.py --test ppl --quant tensor --dtype fp8 --real
python experiments.py --test ppl --quant row --dtype fp8 --real
python experiments.py --test ppl --quant group --group-size 64 --dtype fp8 --real
python experiments.py --test ppl --quant group --group-size 128 --dtype fp8 --real
python experiments.py --test ppl --quant group --group-size 256 --dtype fp8 --real
python experiments.py --test ppl --quant group --group-size 512 --dtype fp8 --real

# Task 3.2

python experiments.py --test throughput \
    --num-samples 100 --prompt-length 1000 --generate-length 1000 \
    --save throughput_results.jsonl

python experiments.py --test throughput --dtype int8 --quant row \
    --num-samples 100 --prompt-length 1000 --generate-length 1000 \
    --save throughput_results.jsonl

python experiments.py --test throughput --dtype fp8 --quant row \
    --num-samples 100 --prompt-length 1000 --generate-length 1000 \
    --save throughput_results.jsonl

python experiments.py --test throughput --dtype int8 --quant row --real \
    --num-samples 100 --prompt-length 1000 --generate-length 1000 \
    --save throughput_results.jsonl

python experiments.py --test throughput --dtype fp8 --quant row --real \
    --num-samples 100 --prompt-length 1000 --generate-length 1000 \
    --save throughput_results.jsonl


python experiments.py --test matmul \
    --matmul-shape 2048 2048 2048 --num-tests 100 \
    --save matmul_results.jsonl

python experiments.py --test matmul --dtype int8 --quant row \
    --matmul-shape 2048 2048 2048 --num-tests 100 \
    --save matmul_results.jsonl

python experiments.py --test matmul --dtype fp8 --quant row \
    --matmul-shape 2048 2048 2048 --num-tests 100 \
    --save matmul_results.jsonl

python experiments.py --test matmul --dtype int8 --quant row --real \
    --matmul-shape 2048 2048 2048 --num-tests 100 \
    --save matmul_results.jsonl

python experiments.py --test matmul --dtype fp8 --quant row --real \
    --matmul-shape 2048 2048 2048 --num-tests 100 \
    --save matmul_results.jsonl

# Task 3.3

python experiments.py --test mmlu --quant smooth --dtype int8 --real
python experiments.py --test mmlu --quant smooth --dtype fp8 --real
python experiments.py --test ppl --quant smooth --dtype int8 --real
python experiments.py --test ppl --quant smooth --dtype fp8 --real

python experiments.py --test throughput --quant smooth --dtype int8 --real \
    --num-samples 100 --prompt-length 1000 --generate-length 1000 \
    --save throughput_results.jsonl

python experiments.py --test throughput --quant smooth --dtype fp8 --real \
    --num-samples 100 --prompt-length 1000 --generate-length 1000 \
    --save throughput_results.jsonl

python experiments.py --test matmul --quant smooth --dtype int8 --real \
    --matmul-shape 2048 2048 2048 --num-tests 100 \
    --save matmul_results.jsonl

python experiments.py --test matmul --quant smooth --dtype fp8 --real \
    --matmul-shape 2048 2048 2048 --num-tests 100 \
    --save matmul_results.jsonl
