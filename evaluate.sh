for dataset in air-pollution microsoft-stock wind-power-generation household_power_consumption_hourly_clean QPS_clean sales_clean; do
    python -m evaluation --dataset $dataset --horizon -1 --model NHITS --experiment run_1
done
