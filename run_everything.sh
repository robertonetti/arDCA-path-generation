#!/bin/bash
# Script per eseguire arDCA su diversi dataset e su diverse coppie di (reg_J, reg_h).
# Ogni run salva il log in "run_regJ<val>_regH<val>.log" nella cartella di output.

# Array dei file di training
train_files=(
                # "generated_data/CM/train_validation/10e1_train.fasta"
                # "generated_data/CM/train_validation/10e2_train.fasta"
                # "generated_data/CM/train_validation/10e3_train.fasta"
                # "generated_data/CM/train_validation/10e4_train.fasta"
                # "generated_data/CM/train_validation/10e5_train.fasta"
                # "generated_data/CM/train_validation/10e6_train.fasta"

                # "generated_data/PF00072_2nd/train_validation/10e1_train.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e2_train.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e3_train.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e4_train.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e5_train.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e6_train.fasta"
)

# Array dei file di test
test_files=(
                # "generated_data/CM/train_validation/10e1_validation.fasta"
                # "generated_data/CM/train_validation/10e2_validation.fasta"
                # "generated_data/CM/train_validation/10e3_validation.fasta"
                # "generated_data/CM/train_validation/10e4_validation.fasta"
                # "generated_data/CM/train_validation/10e5_validation.fasta"
                # "generated_data/CM/train_validation/10e6_validation.fasta"

              
                # "generated_data/PF00072_2nd/train_validation/10e1_validation.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e2_validation.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e3_validation.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e4_validation.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e5_validation.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e6_validation.fasta"
 )

# Array dei path di base di output (senza valori di regolarizzazione)
base_output_paths=(
                # "models_train_val/CM/1cond/10e1"
                # "models_train_val/CM/1cond/10e2"
                # "models_train_val/CM/1cond/10e3"
                # "models_train_val/CM/1cond/10e4"
                # "models_train_val/CM/1cond/10e5"
                # "models_train_val/CM/1cond/10e6"

                # "models_train_val/PF00072_2nd/1cond/10e1"
                # "models_train_val/PF00072_2nd/1cond/10e2"
                # "models_train_val/PF00072_2nd/1cond/10e3"
                # "models_train_val/PF00072_2nd/1cond/10e4"
                # "models_train_val/PF00072_2nd/1cond/10e5"
                # "models_train_val/PF00072_2nd/1cond/10e6"
)

# Array dei valori di reg_J da provare (e coppie reg_h corrispondenti)
regJs=(
    # "1e-5"
    "1e-4"
    # "1e-3"
)
regHs=(
    # "1e-7"
    "1e-6"
    # "1e-5"
)

# Controllo lunghezza array
if [ ${#train_files[@]} -ne ${#test_files[@]} ] || [ ${#train_files[@]} -ne ${#base_output_paths[@]} ]; then
    echo "Errore: gli array train_files, test_files e base_output_paths devono avere la stessa lunghezza."
    exit 1
fi
if [ ${#regJs[@]} -ne ${#regHs[@]} ]; then
    echo "Errore: gli array regJs e regHs devono avere la stessa lunghezza."
    exit 1
fi

# Loop sui dataset
for i in "${!train_files[@]}"; do
    train_file="${train_files[$i]}"
    test_file="${test_files[$i]}"
    base_output="${base_output_paths[$i]}"

    # Loop sui parametri di regolarizzazione a coppie
    for k in "${!regJs[@]}"; do
        regJ="${regJs[$k]}"
        regH="${regHs[$k]}"

        # Costruisco il path di output includendo i parametri
        output_folder="${base_output}_rJ${regJ}_rH${regH}"
        mkdir -p "$output_folder"

        echo "-------------------------------------------------------------------------------------------------------"
        echo "Run dataset $((i+1)) con reg_J=${regJ}, reg_h=${regH}"
        echo ""
        echo "  Train: ${train_file}"
        echo "  Test:  ${test_file}"
        echo "  Output: ${output_folder}"
        echo "-------------------------------------------------------------------------------------------------------"

        # Esegue arDCA e salva stdout+stderr in run_regJ<val>_regH<val>.log
        arDCA_paths train -d "$train_file" \
                    -o "$output_folder" \
                    --nepochs 100000 \
                    --data_test "$test_file" \
                    --no_reweighting \
                    --alphabet "ACDEFGHIKLMNPQRSTVWY-" \
                    --reg_J "$regJ" \
                    --reg_h "$regH" \
                    --mode "second" \
                    --no_entropic_order \
                    --lr 0.005 \
            2>&1 | tee "${output_folder}/run_regJ${regJ}_regH${regH}.log"

        #   --batch_size 85001 \  --path_graph "graphs/evolution_PF00014/graph_only_i_to_i2.pth" \  --no_entropic_order \



        # Verifica l'esito dell'ultima chiamata arDCA
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Errore durante la run dataset $((i+1)), reg_J=${regJ}, reg_h=${regH}. Uscita dallo script."
            exit 1
        fi

    done
done

echo "Tutte le run sono state completate con successo."





#!/bin/bash
# Script per eseguire arDCA su diversi dataset e su diverse coppie di (reg_J, reg_h).
# Ogni run salva il log in "run_regJ<val>_regH<val>.log" nella cartella di output.

# Array dei file di training
train_files=(
                # "generated_data/CM/train_validation/10e1_train.fasta"
                # "generated_data/CM/train_validation/10e2_train.fasta"
                # "generated_data/CM/train_validation/10e3_train.fasta"
                # "generated_data/CM/train_validation/10e4_train.fasta"
                # "generated_data/CM/train_validation/10e5_train.fasta"
                # "generated_data/CM/train_validation/10e6_train.fasta"

                # "generated_data/PF00072_2nd/train_validation/10e1_train.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e2_train.fasta"
                "generated_data/PF00072_2nd/train_validation/10e3_train.fasta"
                "generated_data/PF00072_2nd/train_validation/10e4_train.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e5_train.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e6_train.fasta"

)

# Array dei file di test
test_files=(
                # "generated_data/CM/train_validation/10e1_validation.fasta"
                # "generated_data/CM/train_validation/10e2_validation.fasta"
                # "generated_data/CM/train_validation/10e3_validation.fasta"
                # "generated_data/CM/train_validation/10e4_validation.fasta"
                # "generated_data/CM/train_validation/10e5_validation.fasta"
                # "generated_data/CM/train_validation/10e6_validation.fasta"


                # "generated_data/PF00072_2nd/train_validation/10e1_validation.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e2_validation.fasta"
                "generated_data/PF00072_2nd/train_validation/10e3_validation.fasta"
                "generated_data/PF00072_2nd/train_validation/10e4_validation.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e5_validation.fasta"
                # "generated_data/PF00072_2nd/train_validation/10e6_validation.fasta"
 )

# Array dei path di base di output (senza valori di regolarizzazione)
base_output_paths=(
                # "models_train_val/CM/10e1_long_run"
                # "models_train_val/CM/10e2_long_run"
                # "models_train_val/CM/10e3_long_run"
                # "models_train_val/CM/10e4_long_run"
                # "models_train_val/CM/10e5_long_run"
                # "models_train_val/CM/10e6_long_run"


                # "models_train_val/PF00072_2nd/10e1"
                # "models_train_val/PF00072_2nd/10e2"
                "models_train_val/PF00072_2nd/10e3_long_run"
                "models_train_val/PF00072_2nd/10e4_long_run"
                # "models_train_val/PF00072_2nd/10e5_long_run"
                # "models_train_val/PF00072_2nd/10e6_long_run"
)

# Array dei valori di reg_J da provare (e coppie reg_h corrispondenti)
regJs=(
    # "1e-5"
    "1e-4"
    # "1e-3"
)
regHs=(
    # "1e-7"
    "1e-6"
    # "1e-5"
)

# Controllo lunghezza array
if [ ${#train_files[@]} -ne ${#test_files[@]} ] || [ ${#train_files[@]} -ne ${#base_output_paths[@]} ]; then
    echo "Errore: gli array train_files, test_files e base_output_paths devono avere la stessa lunghezza."
    exit 1
fi
if [ ${#regJs[@]} -ne ${#regHs[@]} ]; then
    echo "Errore: gli array regJs e regHs devono avere la stessa lunghezza."
    exit 1
fi

# Loop sui dataset
for i in "${!train_files[@]}"; do
    train_file="${train_files[$i]}"
    test_file="${test_files[$i]}"
    base_output="${base_output_paths[$i]}"

    # Loop sui parametri di regolarizzazione a coppie
    for k in "${!regJs[@]}"; do
        regJ="${regJs[$k]}"
        regH="${regHs[$k]}"

        # Costruisco il path di output includendo i parametri
        output_folder="${base_output}_rJ${regJ}_rH${regH}"
        mkdir -p "$output_folder"

        echo "-------------------------------------------------------------------------------------------------------"
        echo "Run dataset $((i+1)) con reg_J=${regJ}, reg_h=${regH}"
        echo ""
        echo "  Train: ${train_file}"
        echo "  Test:  ${test_file}"
        echo "  Output: ${output_folder}"
        echo "-------------------------------------------------------------------------------------------------------"

        # Esegue arDCA e salva stdout+stderr in run_regJ<val>_regH<val>.log
        arDCA_paths train -d "$train_file" \
                    -o "$output_folder" \
                    --nepochs 100000 \
                    --data_test "$test_file" \
                    --no_reweighting \
                    --alphabet "ACDEFGHIKLMNPQRSTVWY-" \
                    --reg_J "$regJ" \
                    --reg_h "$regH" \
                    --mode "third" \
                    --no_entropic_order \
                    --lr 0.005 \
            2>&1 | tee "${output_folder}/run_regJ${regJ}_regH${regH}.log"

        #   --batch_size 85001 \  --path_graph "graphs/evolution_PF00014/graph_only_i_to_i2.pth" \  --no_entropic_order \



        # Verifica l'esito dell'ultima chiamata arDCA
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Errore durante la run dataset $((i+1)), reg_J=${regJ}, reg_h=${regH}. Uscita dallo script."
            exit 1
        fi

    done
done

echo "Tutte le run sono state completate con successo."









#!/bin/bash
# Script per eseguire arDCA su diversi dataset e su diverse coppie di (reg_J, reg_h).
# Ogni run salva il log in "run_regJ<val>_regH<val>.log" nella cartella di output.

# Array dei file di training
train_files=(

                # "generated_data/betalactamase/train_validation/10e1_train.fasta"
                # "generated_data/betalactamase/train_validation/10e2_train.fasta"
                # "generated_data/betalactamase/train_validation/10e3_train.fasta"
                # "generated_data/betalactamase/train_validation/10e4_train.fasta"
                # "generated_data/betalactamase/train_validation/10e5_train.fasta"
                # "generated_data/betalactamase/train_validation/10e6_train.fasta"

)

# Array dei file di test
test_files=(

                # "generated_data/betalactamase/train_validation/10e1_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e2_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e3_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e4_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e5_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e6_validation.fasta"

 )

# Array dei path di base di output (senza valori di regolarizzazione)
base_output_paths=(
                # "models_train_val/betalactamase/10e1"
                # "models_train_val/betalactamase/10e2"
                # "models_train_val/betalactamase/10e3"
                # "models_train_val/betalactamase/10e4"
                # "models_train_val/betalactamase/10e5"
                # "models_train_val/betalactamase/10e6"

)

# Array dei valori di reg_J da provare (e coppie reg_h corrispondenti)
regJs=(
    # "1e-5"
    # "1e-4"
    "1e-3"
)
regHs=(
    # "1e-7"
    # "5e-6"
    "1e-5"
)

# Controllo lunghezza array
if [ ${#train_files[@]} -ne ${#test_files[@]} ] || [ ${#train_files[@]} -ne ${#base_output_paths[@]} ]; then
    echo "Errore: gli array train_files, test_files e base_output_paths devono avere la stessa lunghezza."
    exit 1
fi
if [ ${#regJs[@]} -ne ${#regHs[@]} ]; then
    echo "Errore: gli array regJs e regHs devono avere la stessa lunghezza."
    exit 1
fi

# Loop sui dataset
for i in "${!train_files[@]}"; do
    train_file="${train_files[$i]}"
    test_file="${test_files[$i]}"
    base_output="${base_output_paths[$i]}"

    # Loop sui parametri di regolarizzazione a coppie
    for k in "${!regJs[@]}"; do
        regJ="${regJs[$k]}"
        regH="${regHs[$k]}"

        # Costruisco il path di output includendo i parametri
        output_folder="${base_output}_rJ${regJ}_rH${regH}"
        mkdir -p "$output_folder"

        echo "-------------------------------------------------------------------------------------------------------"
        echo "Run dataset $((i+1)) con reg_J=${regJ}, reg_h=${regH}"
        echo ""
        echo "  Train: ${train_file}"
        echo "  Test:  ${test_file}"
        echo "  Output: ${output_folder}"
        echo "-------------------------------------------------------------------------------------------------------"

        # Esegue arDCA e salva stdout+stderr in run_regJ<val>_regH<val>.log
        arDCA_paths train -d "$train_file" \
                    -o "$output_folder" \
                    --nepochs 100000 \
                    --data_test "$test_file" \
                    --no_reweighting \
                    --alphabet "ACDEFGHIKLMNPQRSTVWY-" \
                    --reg_J "$regJ" \
                    --reg_h "$regH" \
                    --mode "third" \
                    --no_entropic_order \
                    --lr 0.005 \
                    --batch_size 14000 \
            2>&1 | tee "${output_folder}/run_regJ${regJ}_regH${regH}.log"

        #   --batch_size 85001 \  --path_graph "graphs/evolution_PF00014/graph_only_i_to_i2.pth" \  --no_entropic_order \



        # Verifica l'esito dell'ultima chiamata arDCA
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Errore durante la run dataset $((i+1)), reg_J=${regJ}, reg_h=${regH}. Uscita dallo script."
            exit 1
        fi

    done
done

echo "Tutte le run sono state completate con successo."
















#!/bin/bash
# Script per eseguire arDCA su diversi dataset e su diverse coppie di (reg_J, reg_h).
# Ogni run salva il log in "run_regJ<val>_regH<val>.log" nella cartella di output.

# Array dei file di training
train_files=(
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_5_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_10_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_15_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_20_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_25_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_30_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_35_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_40_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_45_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_50_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_55_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_60_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_65_train_large.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_70_train_large.fasta"


    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_5_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_10_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_15_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_20_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_25_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_30_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_35_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_40_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_45_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_50_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_55_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_60_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_65_train.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_70_train.fasta"

)

# Array dei file di test
test_files=(

    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_5_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_10_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_15_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_20_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_25_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_30_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_35_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_40_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_45_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_50_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_55_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_60_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_65_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered/distance_70_validation.fasta"

    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_5_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_10_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_15_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_20_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_25_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_30_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_35_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_40_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_45_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_50_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_55_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_60_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_65_validation.fasta"
    # "generated_data/CM/reorganized_per_distance/new_filtering/non_filtered_no_cap/distance_70_validation.fasta"


)
# Array dei path di base di output (senza valori di regolarizzazione)
base_output_paths=(

    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_5"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_10"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_15"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_20"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_25"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_30"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_35"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_40"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_45"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_50"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_55"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_60"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_65"
    # "models_train_val/CM/trained_per_distance/non_filtered/model_d_upto_70"

    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_5"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_10"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_15"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_20"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_25"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_30"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_35"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_40"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_45"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_50"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_55"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_60"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_65"
    # "models_train_val/CM/trained_per_distance/non_filtered_no_cap/model_d_upto_70"


)

# Array dei valori di reg_J da provare (e coppie reg_h corrispondenti)
regJs=(
    # "5e-5"
    "1e-4"
    # "5e-3"
)
regHs=(
    # "5e-7"
    "1e-6"
    # "5e-5"
)

# Controllo lunghezza array
if [ ${#train_files[@]} -ne ${#test_files[@]} ] || [ ${#train_files[@]} -ne ${#base_output_paths[@]} ]; then
    echo "Errore: gli array train_files, test_files e base_output_paths devono avere la stessa lunghezza."
    exit 1
fi
if [ ${#regJs[@]} -ne ${#regHs[@]} ]; then
    echo "Errore: gli array regJs e regHs devono avere la stessa lunghezza."
    exit 1
fi

# Loop sui dataset
for i in "${!train_files[@]}"; do
    train_file="${train_files[$i]}"
    test_file="${test_files[$i]}"
    base_output="${base_output_paths[$i]}"

    # Loop sui parametri di regolarizzazione a coppie
    for k in "${!regJs[@]}"; do
        regJ="${regJs[$k]}"
        regH="${regHs[$k]}"

        # Costruisco il path di output includendo i parametri
        output_folder="${base_output}_rJ${regJ}_rH${regH}"
        mkdir -p "$output_folder"

        echo "-------------------------------------------------------------------------------------------------------"
        echo "Run dataset $((i+1)) con reg_J=${regJ}, reg_h=${regH}"
        echo ""
        echo "  Train: ${train_file}"
        echo "  Test:  ${test_file}"
        echo "  Output: ${output_folder}"
        echo "-------------------------------------------------------------------------------------------------------"

        # Esegue arDCA e salva stdout+stderr in run_regJ<val>_regH<val>.log
        arDCA_paths train -d "$train_file" \
                    -o "$output_folder" \
                    --nepochs 100000 \
                    --data_test "$test_file" \
                    --no_reweighting \
                    --alphabet "ACDEFGHIKLMNPQRSTVWY-" \
                    --reg_J "$regJ" \
                    --reg_h "$regH" \
                    --mode "third" \
                    --no_entropic_order \
                    --lr 0.005 \
            2>&1 | tee "${output_folder}/run_regJ${regJ}_regH${regH}.log"

        #   --batch_size 85001 \  --path_graph "graphs/evolution_PF00014/graph_only_i_to_i2.pth" \  --no_entropic_order \


        # Verifica l'esito dell'ultima chiamata arDCA
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Errore durante la run dataset $((i+1)), reg_J=${regJ}, reg_h=${regH}. Uscita dallo script."
            exit 1
        fi

    done
done

echo "Tutte le run sono state completate con successo."
















#!/bin/bash
# Script per eseguire arDCA su diversi dataset e su diverse coppie di (reg_J, reg_h).
# Ogni run salva il log in "run_regJ<val>_regH<val>.log" nella cartella di output.

# Array dei file di training
train_files=(
                
                # "generated_data/betalactamase/train_validation/10e1_train.fasta"
                # "generated_data/betalactamase/train_validation/10e2_train.fasta"
                # "generated_data/betalactamase/train_validation/10e3_train.fasta"
                # "generated_data/betalactamase/train_validation/10e4_train.fasta"
                # "generated_data/betalactamase/train_validation/10e5_train.fasta"
                # "generated_data/betalactamase/train_validation/10e6_train.fasta"

)

# Array dei file di test
test_files=(
               

                # "generated_data/betalactamase/train_validation/10e1_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e2_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e3_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e4_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e5_validation.fasta"
                # "generated_data/betalactamase/train_validation/10e6_validation.fasta"

)

# Array dei path di base di output (senza valori di regolarizzazione)
base_output_paths=(
    

                # "models_train_val/betalactamase/1cond/10e1"
                # "models_train_val/betalactamase/1cond/10e2"
                # "models_train_val/betalactamase/1cond/10e3"
                # "models_train_val/betalactamase/1cond/10e4"
                # "models_train_val/betalactamase/1cond/10e5"
                # "models_train_val/betalactamase/1cond/10e6"
)

# Array dei valori di reg_J da provare (e coppie reg_h corrispondenti)
regJs=(
    # "1e-5"
    # "1e-4"
    "1e-3"
)
regHs=(
    # "1e-7"
    # "1e-6"
    "1e-5"
)

# Controllo lunghezza array
if [ ${#train_files[@]} -ne ${#test_files[@]} ] || [ ${#train_files[@]} -ne ${#base_output_paths[@]} ]; then
    echo "Errore: gli array train_files, test_files e base_output_paths devono avere la stessa lunghezza."
    exit 1
fi
if [ ${#regJs[@]} -ne ${#regHs[@]} ]; then
    echo "Errore: gli array regJs e regHs devono avere la stessa lunghezza."
    exit 1
fi

# Loop sui dataset
for i in "${!train_files[@]}"; do
    train_file="${train_files[$i]}"
    test_file="${test_files[$i]}"
    base_output="${base_output_paths[$i]}"

    # Loop sui parametri di regolarizzazione a coppie
    for k in "${!regJs[@]}"; do
        regJ="${regJs[$k]}"
        regH="${regHs[$k]}"

        # Costruisco il path di output includendo i parametri
        output_folder="${base_output}_rJ${regJ}_rH${regH}"
        mkdir -p "$output_folder"

        echo "-------------------------------------------------------------------------------------------------------"
        echo "Run dataset $((i+1)) con reg_J=${regJ}, reg_h=${regH}"
        echo ""
        echo "  Train: ${train_file}"
        echo "  Test:  ${test_file}"
        echo "  Output: ${output_folder}"
        echo "-------------------------------------------------------------------------------------------------------"

        # Esegue arDCA e salva stdout+stderr in run_regJ<val>_regH<val>.log
        arDCA_paths train -d "$train_file" \
                    -o "$output_folder" \
                    --nepochs 100000 \
                    --data_test "$test_file" \
                    --no_reweighting \
                    --alphabet "ACDEFGHIKLMNPQRSTVWY-" \
                    --reg_J "$regJ" \
                    --reg_h "$regH" \
                    --mode "second" \
                    --no_entropic_order \
                    --lr 0.005 \
                    --batch_size 14000 \
            2>&1 | tee "${output_folder}/run_regJ${regJ}_regH${regH}.log"

        #   --batch_size 85001 \  --path_graph "graphs/evolution_PF00014/graph_only_i_to_i2.pth" \  --no_entropic_order \



        # Verifica l'esito dell'ultima chiamata arDCA
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Errore durante la run dataset $((i+1)), reg_J=${regJ}, reg_h=${regH}. Uscita dallo script."
            exit 1
        fi

    done
done