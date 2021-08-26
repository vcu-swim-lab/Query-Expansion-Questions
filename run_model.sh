MODEL_SCRIPT_DIR=models
TEST_SCRIPT_DIR=tests
EMB_DIR=embeddings
DATASET_DIR=dataset
RESULT_DIR=outputs

python $TEST_SCRIPT_DIR/test_model.py \
        --embeddings $EMB_DIR/vectors_pad.txt \
        --models $MODEL_SCRIPT_DIR/ \
        --dataset $DATASET_DIR/ \
        --templates $DATASET_DIR/templates.csv \
        --test-dataset $DATASET_DIR/test_dataset.csv \
        --all-dataset $DATASET_DIR/gen-queries.csv \
        --results $RESULT_DIR/ \
        --batch-size 64 \
        --n-epochs 10 \
        --filler-size '2 3' \
