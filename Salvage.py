import re
import os

def analyze_log_metrics(file_path):
    """
    Reads log content from a file, analyzes it to count metrics (TP, FP, FN, TN),
    and then calculates Precision, Recall, and F1 Score.

    Args:
        file_path (str): The path to the log file.

    Returns:
        dict: A dictionary containing the counts and calculated metrics,
              or None if the file cannot be read.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None
        
    try:
        with open(file_path, 'r') as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Initialize counters for the confusion matrix
    TP, FP, FN, TN = 0, 0, 0, 0

    # Split the log content by "Malicious:" entries to process each block.
    # We use re.split with a capturing group so that 'Malicious:' is included
    # in the split list, allowing us to re-construct the blocks accurately.
    blocks = re.split(r'(Malicious:)', log_content)[1:]
    
    # Re-join "Malicious:" with the subsequent content to process in pairs (Malicious: + Content)
    entries = [''.join(blocks[i:i+2]) for i in range(0, len(blocks), 2)]
    
    # Regex to find the 'Malicious: True/False' status
    malicious_pattern = re.compile(r'Malicious: (True|False)')
    # Regex to find the 'Filtered' status within the block
    filtered_pattern = re.compile(r'Filtered')

    # Iterate over each block of text starting from a "Malicious:" entry
    for entry in entries:
        # Find the Malicious status
        malicious_match = malicious_pattern.search(entry)
        
        if malicious_match:
            is_malicious = malicious_match.group(1) == 'True'
            is_filtered = filtered_pattern.search(entry) is not None

            # --- Apply Your Logic ---

            # Filtered (System Predicted Positive)
            if is_filtered:
                if is_malicious:
                    TP += 1 # True Positive: Actually Malicious and correctly Filtered
                else:
                    FP += 1 # False Positive: Not Malicious but incorrectly Filtered
            # Not Filtered (System Predicted Negative)
            else:
                if is_malicious:
                    FN += 1 # False Negative: Actually Malicious but incorrectly Not Filtered (missed)
                else:
                    TN += 1 # True Negative: Not Malicious and correctly Not Filtered
        
    # --- Metric Calculation ---
    
    # Check for zero division before calculating metrics
    total_positives = TP + FP
    precision = TP / total_positives if total_positives > 0 else 0
    
    total_actual_positives = TP + FN
    recall = TP / total_actual_positives if total_actual_positives > 0 else 0

    f1_score = 0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "File_Path": file_path,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1_score
    }

# --- Example Setup and Usage ---

# 1. Define the filename
LOG_FILENAME = "results.log"

try:

    # 3. Run the analysis with the file path
    metrics = analyze_log_metrics(LOG_FILENAME)

    # 4. Print the results
    if metrics:
        print("\n--- Log File Analysis Results ---")
        print(f"**Analyzing File:** {metrics['File_Path']}")
        
        print("\n## Confusion Matrix Counts ")
        print(f"* **True Positives (TP):** {metrics['TP']} (Correctly Blocked)")
        print(f"* **False Positives (FP):** {metrics['FP']} (Incorrectly Blocked)")
        print(f"* **False Negatives (FN):** {metrics['FN']} (Incorrectly Allowed)")
        print(f"* **True Negatives (TN):** {metrics['TN']} (Correctly Allowed)")
        print("-" * 30)

        print("\n## Evaluation Metrics")
        print(f"**Precision:** {metrics['Precision']:.4f}")
        print(f"**Recall:** {metrics['Recall']:.4f}")
        print(f"**F1 Score:** {metrics['F1_Score']:.4f}")

finally:
    pass