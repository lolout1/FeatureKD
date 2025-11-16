#!/bin/bash

# Train the IMU student model with accelerometer + gyroscope input.
imu_config="./config/smartfallmm/imu_student.yaml"

if [ -f "$imu_config" ]; then
    echo "========================================="
    echo "IMU Fall Detection Training"
    echo "========================================="
    echo "Reading YAML config: $imu_config"
    python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])))" "$imu_config"
    echo "========================================="

    sensor_value=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('dataset_args', {}).get('sensors', '')[0])" "$imu_config")
    use_skeleton=$(python3 -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(str(cfg.get('dataset_args', {}).get('use_skeleton', True)))" "$imu_config")
    echo "Sensor: $sensor_value"
    echo "Modalities: Accelerometer + Gyroscope (6-channel IMU)"
    echo "Age Group: Young participants only"
    if [[ "$use_skeleton" == "True" ]]; then
        echo "Skeleton alignment: enabled"
    else
        echo "Skeleton alignment: disabled (IMU-only)"
    fi
    echo "========================================="
else
    echo "Config file not found: $imu_config"
    exit 1
fi

imu_weights="imu_student_best"
work_dir="exps/smartfall_imu/student/${sensor_value}_acc_gyro_6ch_young"
result_file="result.txt"

device=0

echo ""
echo "========================================="
echo "Starting Training Phase"
echo "========================================="
echo "Model: IMUTransformer (6-channel: ax, ay, az, gx, gy, gz)"
echo "Output Directory: $work_dir"
echo "Weight Name: $imu_weights"
echo "Device: GPU $device"
echo "========================================="

python3 main.py \
    --config "$imu_config" \
    --work-dir "$work_dir" \
    --model-saved-name "$imu_weights" \
    --device "$device" \
    --include-val True \
    --phase 'train'

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Training completed successfully!"
    echo "========================================="
    echo "Model weights saved to: $work_dir/$imu_weights"
    echo ""

    read -p "Do you want to run the testing phase? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "========================================="
        echo "Starting Testing Phase"
        echo "========================================="

        python3 main.py \
            --config "$imu_config" \
            --work-dir "$work_dir" \
            --weights "$work_dir/$imu_weights" \
            --device "$device" \
            --phase 'test'

        if [ $? -eq 0 ]; then
            echo ""
            echo "========================================="
            echo "Testing completed successfully!"
            echo "Results saved to: $work_dir"
            echo "========================================="
        else
            echo "Testing failed!"
            exit 1
        fi
    fi
else
    echo "Training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "Script completed"
echo "========================================="
echo ""
echo "Summary:"
echo "  - Model: IMUTransformer"
echo "  - Sensor: $sensor_value"
echo "  - Input: 6 channels (accelerometer + gyroscope)"
echo "  - Age Group: Young participants"
echo "  - Filtering: Enabled (Butterworth, cutoff=5.5 Hz)"
echo "  - Window Size: 128 samples (suitable for Android real-time)"
if [[ "$use_skeleton" == "True" ]]; then
    echo "  - Skeleton Alignment: Enabled"
else
    echo "  - Skeleton Alignment: Disabled"
fi
echo "  - Work Directory: $work_dir"
echo ""
echo "To test different sensors, modify 'sensors' in the config file:"
echo "  - watch"
echo "  - phone"
echo "  - meta_wrist"
echo "  - meta_hip"
echo ""
echo "To adjust the model for less overfitting:"
echo "  1. Increase dropout (currently 0.5)"
echo "  2. Reduce embed_dim (currently 32)"
echo "  3. Reduce num_layers (currently 2)"
echo "  4. Use the IMUTransformerLight model variant"
echo ""
echo "========================================="
