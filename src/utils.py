import numpy as np


def merge_frame_sequences(sequences, limit=None):
    # Step 1: Replace None frames with the previous valid frame
    sequences_processed = []
    for sequence in sequences:
        # Find the first valid frame
        frame_valid_first = None
        for frame in sequence:
            if frame is not None:
                frame_valid_first = frame
                break
        if frame_valid_first is None:
            # raise ValueError("All frames in the sequence are None!")
            continue  # skip empty sequence
        # Replace Nones with previous valid
        sequence_ndarrays = []
        frame_valid_last = frame_valid_first
        for frame in sequence:
            if frame is not None:
                sequence_ndarrays.append(frame)
                frame_valid_last = frame
            else:
                sequence_ndarrays.append(frame_valid_last)
        sequences_processed.append(sequence_ndarrays)

    # Step 2: Determine the maximum sequence length
    max_length = max(len(s) for s in sequences_processed)

    # Step 3: Pad each sequence to max_length by repeating the last frame
    sequences_padded = []
    for sequence in sequences_processed:
        sequence_ndarray = np.array(sequence)
        n = sequence_ndarray.shape[0]
        if n < max_length:
            pad_size = max_length - n
            frame_last = sequence_ndarray[-1]
            # Repeat the last frame pad_size times
            sequence_pad = np.repeat(frame_last[np.newaxis, ...], pad_size, axis=0)
            sequence_pad = np.concatenate([sequence_ndarray, sequence_pad], axis=0)
        else:
            sequence_pad = sequence_ndarray
        sequences_padded.append(sequence_pad[:limit])

    # Step 4: Stack trajectories and compute the mean
    sequences_stacked = np.stack(sequences_padded, axis=0)
    sequence_mean = np.mean(sequences_stacked, axis=0)

    return sequence_mean
