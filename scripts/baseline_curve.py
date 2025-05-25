import json
import numpy as np

def load_sequences(json_path):
    """Load sequences from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Assume data is a list of dicts with 'sequence' key containing list of feature vectors
    sequences = [np.array(sample['sequence']) for sample in data]
    return sequences

def compute_masked_average_baseline(sequences):
    """Compute average feature values per frame, ignoring zero frames."""
    seq_len = max(seq.shape[0] for seq in sequences)
    n_features = sequences[0].shape[1]
    
    # Pad all sequences to the max length with zeros if needed
    padded_seqs = []
    for seq in sequences:
        if seq.shape[0] < seq_len:
            padding = np.zeros((seq_len - seq.shape[0], n_features))
            seq_padded = np.vstack([seq, padding])
        else:
            seq_padded = seq
        padded_seqs.append(seq_padded)
    
    stacked = np.stack(padded_seqs)  # shape: (num_samples, seq_len, n_features)
    
    # Mask: True where any feature is non-zero
    mask = np.any(stacked != 0, axis=2)  # shape: (num_samples, seq_len)
    
    sum_arr = np.zeros((seq_len, n_features))
    count_arr = np.zeros((seq_len, n_features))
    
    for sample_idx in range(stacked.shape[0]):
        for frame_idx in range(seq_len):
            if mask[sample_idx, frame_idx]:
                sum_arr[frame_idx] += stacked[sample_idx, frame_idx]
                count_arr[frame_idx] += 1
    
    # Compute average avoiding division by zero
    avg_arr = np.divide(sum_arr, count_arr, out=np.zeros_like(sum_arr), where=count_arr!=0)
    return avg_arr

def save_baseline_json(baseline_arr, feature_names, output_path):
    """Save baseline as JSON with feature names as keys."""
    baseline_dict = {feature_names[i]: baseline_arr[:, i].tolist() for i in range(len(feature_names))}
    with open(output_path, 'w') as f:
        json.dump(baseline_dict, f, indent=2)

def main():
    input_json = 'good_bowling_sequences.json'  # Your JSON file path with good samples
    output_json = 'baseline_good_curve.json'
    
    # Feature order must match your sequence vector
    feature_names = [
        "left_elbow_angle",
        "right_elbow_angle",
        "shoulder_alignment",
        "jump_height",
        "hip_shoulder_separation",
        "head_stability"
    ]
    
    sequences = load_sequences(input_json)
    baseline_arr = compute_masked_average_baseline(sequences)
    save_baseline_json(baseline_arr, feature_names, output_json)
    print(f"Baseline saved to {output_json}")

if __name__ == "__main__":
    main()
