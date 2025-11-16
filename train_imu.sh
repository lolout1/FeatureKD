#!/bin/bash

#######################################
# Training Script for IMU-based Fall Detection
# Combined Accelerometer + Gyroscope Model
# Young Participants Only
#######################################

# Configuration
imu_config="./config/smartfallmm/imu_student.yaml"

# Check if config file exists
if [ -f "$imu_config" ]; then
    echo "========================================="
    echo "IMU Fall Detection Training"
    echo "========================================="
    echo "Reading YAML config: $imu_config"
    python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])))" "$imu_config"
    echo "========================================="

    # Extract sensor value from YAML config
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

# Directory and weight configuration
imu_weights="imu_student_best"
work_dir="exps/smartfall_imu/student/${sensor_value}_acc_gyro_6ch_young"
result_file="result.txt"

# Device configuration (adjust based on your GPU availability)
device=0

# Training Phase
echo ""
echo "========================================="
echo "Starting Training Phase"
echo "========================================="
echo "Model: IMUTransformer (6-channel: ax, ay, az, gx, gy, gz)"
echo "Output Directory: $work_dir"
echo "Weight Name: $imu_weights"
echo "Device: GPU $device"
echo "========================================="

# Train the IMU student model without knowledge distillation
# This trains on accelerometer + gyroscope data for young participants
python3 main.py \
    --config "$imu_config" \
    --work-dir "$work_dir" \
    --model-saved-name "$imu_weights" \
    --device "$device" \
    --include-val True \
    --phase 'train'

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Training completed successfully!"
    echo "========================================="
    echo "Model weights saved to: $work_dir/$imu_weights"
    echo ""

    # Optional: Run testing phase after training
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

# Additional training scripts for different configurations
# Uncomment and modify as needed:

# # Train with phone sensor
# # python3 main.py --config ./config/smartfallmm/imu_student.yaml --work-dir exps/smartfall_imu/student/phone_acc_gyro_6ch_young --model-saved-name imu_student_best --device 0 --include-val True

# # Train with meta_wrist sensor
# # python3 main.py --config ./config/smartfallmm/imu_student.yaml --work-dir exps/smartfall_imu/student/meta_wrist_acc_gyro_6ch_young --model-saved-name imu_student_best --device 0 --include-val True

# # Train with meta_hip sensor
# # python3 main.py --config ./config/smartfallmm/imu_student.yaml --work-dir exps/smartfall_imu/student/meta_hip_acc_gyro_6ch_young --model-saved-name imu_student_best --device 0 --include-val True

# # Test only mode (if model is already trained)
# # python3 main.py --config ./config/smartfallmm/imu_student.yaml --work-dir "$work_dir" --weights "$work_dir/$imu_weights" --device 0 --phase 'test'

# # Knowledge distillation (if you have a trained teacher model)
# # teacher_dir="$HOME/LightHART/exps/smartfall_fall_wokd/teacher/skeleton_with_experimental_meta_hip_again"
# # teacher_weights="spTransformer"
# # python3 distiller.py --config ./config/smartfallmm/distill.yaml --work-dir "$work_dir"  --teacher-weight "$teacher_dir/$teacher_weights" --model-saved-name "$imu_weights" --device 0 --include-val True
