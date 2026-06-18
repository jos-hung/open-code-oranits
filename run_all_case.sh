OUTPUT_FILE="output.txt"

> "$OUTPUT_FILE"

for ((a=10; a<=100; a++)); do
    echo "Running: python run.py -i None -a $a" | tee -a "$OUTPUT_FILE"
    python run.py -i None -a "$a" >> "$OUTPUT_FILE" 2>&1
    echo "--------------------------------------" >> "$OUTPUT_FILE"
done