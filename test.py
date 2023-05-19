import note_seq

from evaluation.metrics import trans_control_changes_to_onset_offset_events, get_pedal_score

ns = note_seq.midi_file_to_note_sequence("/data/lobby/mt3/maestro-v3.0.0/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi")
print(get_pedal_score(ns,ns))