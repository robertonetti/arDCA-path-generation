#!/bin/bash
# Script per eseguire arDCA su diversi dataset e su diverse coppie di (reg_J, reg_h).
# Ogni run salva il log in "run_regJ<val>_regH<val>.log" nella cartella di output.

# Array dei file di training
train_files=(
                "generated_data/PF00072/10e1_train.fasta"
                "generated_data/PF00072/10e2_train.fasta"
                "generated_data/PF00072/10e3_train.fasta"
                "generated_data/PF00072/10e4_train.fasta"
                "generated_data/PF00072/10e5_train.fasta"
                "generated_data/PF00072/10e6_train.fasta"
)

# Array dei file di test
test_files=(
                "generated_data/PF00072/10e1_test.fasta"
                "generated_data/PF00072/10e2_test.fasta"
                "generated_data/PF00072/10e3_test.fasta"
                "generated_data/PF00072/10e4_test.fasta"
                "generated_data/PF00072/10e5_test.fasta"
                "generated_data/PF00072/10e6_test.fasta"
 )

# Array dei path di base di output (senza valori di regolarizzazione)
base_output_paths=(
                "models_evotimescales/PF00072/10e1"
                "models_evotimescales/PF00072/10e2"
                "models_evotimescales/PF00072/10e3"
                "models_evotimescales/PF00072/10e4"
                "models_evotimescales/PF00072/10e5"
                "models_evotimescales/PF00072/10e6"
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
                    --lr 0.01 \
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
