for dataset in air-pollution microsoft-stock wind-power-generation household_power_consumption_hourly_clean QPS_clean sales_clean; do
    python -m nhits_multivariate --hyperopt_max_evals 10 --experiment_id run_1 --dataset $dataset
done