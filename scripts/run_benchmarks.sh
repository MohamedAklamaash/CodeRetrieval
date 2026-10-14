# Runs the three evaluation sets. Lite test and Verified are time-boxed because CPU embedding of every
# historical commit is the bottleneck; instances are processed in a seeded random order, so whatever completes
# is a random subset of the stratified sample. Re-running resumes where a run stopped.
set -u
CR="uv run code-retrieval --config configs/evaluation.toml evaluate"
TEST_HOURS=${TEST_HOURS:-5}
MULTI_HOURS=${MULTI_HOURS:-3}
$CR --dataset lite --split dev --out results/lite-dev
timeout --signal=INT "${TEST_HOURS}h" $CR --dataset lite --split test --sample 120 --seed 13 --out results/lite-test-s120
timeout --signal=INT "${MULTI_HOURS}h" $CR --dataset verified --split test --min-files 2 --sample 50 --seed 13 --out results/verified-multifile-s50
echo "benchmarks finished"
