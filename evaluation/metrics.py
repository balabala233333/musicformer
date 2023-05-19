import collections
import dataclasses
from typing import Tuple, Mapping

import mir_eval
import note_seq
import numpy as np
import pretty_midi
from sklearn.metrics import precision_recall_fscore_support

from constant import PEDAL_EXTEND, _SUSTAIN_ON, _SUSTAIN_OFF


@dataclasses.dataclass
class EvaluationItem:
    name: str
    f1_score: float
    precision_score: float
    recall_score: float


def frame_metrics(ref_pianoroll: np.ndarray,
                  est_pianoroll: np.ndarray,
                  velocity_threshold: int) -> Tuple[float, float, float]:
    if ref_pianoroll.shape[1] > est_pianoroll.shape[1]:
        diff = ref_pianoroll.shape[1] - est_pianoroll.shape[1]
        est_pianoroll = np.pad(est_pianoroll, [(0, 0), (0, diff)], mode='constant')
    elif est_pianoroll.shape[1] > ref_pianoroll.shape[1]:
        diff = est_pianoroll.shape[1] - ref_pianoroll.shape[1]
        ref_pianoroll = np.pad(ref_pianoroll, [(0, 0), (0, diff)], mode='constant')

    # For ref, remove any notes that are too quiet (consistent with Cerberus.)
    ref_frames_bool = ref_pianoroll > velocity_threshold
    # For est, keep all predicted notes.
    est_frames_bool = est_pianoroll > 0

    precision, recall, f1, _ = precision_recall_fscore_support(
        ref_frames_bool.flatten(),
        est_frames_bool.flatten(),
        labels=[True, False])

    return precision[0], recall[0], f1[0]


def get_prettymidi_pianoroll(ns: note_seq.NoteSequence, fps: float,
                             is_drum: bool):
    for note in ns.notes:
        if is_drum or note.end_time - note.start_time < 0.05:
            # Give all drum notes a fixed length, and all others a min length
            note.end_time = note.start_time + 0.05

    pm = note_seq.note_sequence_to_pretty_midi(ns)
    end_time = pm.get_end_time()
    cc = [
        # all sound off
        pretty_midi.ControlChange(number=120, value=0, time=end_time),
        # all notes off
        pretty_midi.ControlChange(number=123, value=0, time=end_time)
    ]
    pm.instruments[0].control_changes = cc
    if is_drum:
        # If inst.is_drum is set, pretty_midi will return an all zero pianoroll.
        for inst in pm.instruments:
            inst.is_drum = False
    pianoroll = pm.get_piano_roll(fs=fps)
    return pianoroll


def calculate_frame_score(target_ns: note_seq.NoteSequence,
                          prediction_ns: note_seq.NoteSequence) -> Tuple[float, float, float]:
    pred_rolls = get_prettymidi_pianoroll(prediction_ns, 62.5, is_drum=False)
    targets_rolls = get_prettymidi_pianoroll(target_ns, 62.5, is_drum=False)
    frame_precision, frame_recall, frame_f1 = frame_metrics(
        targets_rolls, pred_rolls, velocity_threshold=30)
    return EvaluationItem('frame_score', f1_score=frame_f1, precision_score=frame_precision, recall_score=frame_recall)


def calculate_onset_score(target_ns: note_seq.NoteSequence, prediction_ns: note_seq.NoteSequence) -> EvaluationItem:
    est_intervals, est_pitches, est_velocities = (
        note_seq.sequences_lib.sequence_to_valued_intervals(prediction_ns))

    ref_intervals, ref_pitches, ref_velocities = (
        note_seq.sequences_lib.sequence_to_valued_intervals(target_ns))

    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches, offset_ratio=None))
    return EvaluationItem('onset_score', f1_score=f_measure, precision_score=precision, recall_score=recall)


def calculate_onset_and_offset_score(target_ns: note_seq.NoteSequence,
                                     prediction_ns: note_seq.NoteSequence) -> EvaluationItem:
    est_intervals, est_pitches, est_velocities = (
        note_seq.sequences_lib.sequence_to_valued_intervals(prediction_ns))

    ref_intervals, ref_pitches, ref_velocities = (
        note_seq.sequences_lib.sequence_to_valued_intervals(target_ns))

    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05))
    return EvaluationItem('onset_offset_score', f1_score=f_measure, precision_score=precision, recall_score=recall)


def calculate_onset_and_offset_velocity_score(target_ns: note_seq.NoteSequence,
                                              prediction_ns: note_seq.NoteSequence) -> EvaluationItem:
    est_intervals, est_pitches, est_velocities = (
        note_seq.sequences_lib.sequence_to_valued_intervals(prediction_ns))

    ref_intervals, ref_pitches, ref_velocities = (
        note_seq.sequences_lib.sequence_to_valued_intervals(target_ns))
    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            ref_velocities=ref_velocities,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            est_velocities=est_velocities, onset_tolerance=0.05,
            offset_ratio=0.2, offset_min_tolerance=0.05,
            velocity_tolerance=0.1))
    return EvaluationItem('onset_offset_velocity_score', f1_score=f_measure, precision_score=precision,
                          recall_score=recall)


def get_scores(target_ns: note_seq.NoteSequence,
               prediction_ns: note_seq.NoteSequence) -> Mapping[str, EvaluationItem]:
    if PEDAL_EXTEND:
        target_ns = note_seq.apply_sustain_control_changes(target_ns)
    return {
        "onset_score": calculate_onset_score(target_ns, prediction_ns),
        "onset_offset_score": calculate_onset_and_offset_score(target_ns, prediction_ns),
        "onset_offset_velocity_score": calculate_onset_and_offset_velocity_score(target_ns, prediction_ns),
        "frame_score": calculate_frame_score(target_ns, prediction_ns)
    }


def trans_control_changes_to_onset_offset_events(ns: note_seq.NoteSequence, all_pedal: bool = False):
    events = []
    state = collections.defaultdict(lambda: False)
    onset = []
    offset = []
    pitch = []
    intervals = []
    for cc in ns.control_changes:
        if all_pedal == False and cc.control_number != 64:
            continue
        value = cc.control_value
        if value >= 64:
            events.append((cc.time, _SUSTAIN_ON, cc))
        elif value < 64:
            events.append((cc.time, _SUSTAIN_OFF, cc))
    for event in events:
        time, s, cc = event
        if state[cc.control_number] == _SUSTAIN_OFF and s == _SUSTAIN_ON:
            onset.append(time)
            state[cc.control_number] = _SUSTAIN_ON
        elif state[cc.control_number] == _SUSTAIN_ON and s == _SUSTAIN_OFF:
            offset.append(time)
            pitch.append(cc.control_number)
            state[cc.control_number] = _SUSTAIN_OFF
    for k in state.keys():
        if state[k] == _SUSTAIN_ON:
            offset.append(time)
            pitch.append(k)
            state[cc.control_number] = _SUSTAIN_OFF
    for k, v in zip(onset, offset):
        intervals.append((k, v))
    return np.array(intervals), np.array(pitch)


def get_pedal_score(ref_ns: note_seq.NoteSequence, est_ns: note_seq.NoteSequence, all_pedal: bool = False):
    ref_intervals, ref_pitches = trans_control_changes_to_onset_offset_events(ref_ns, all_pedal)
    est_intervals, est_pitches = trans_control_changes_to_onset_offset_events(est_ns, all_pedal)
    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches, offset_ratio=None))
    onset_item = EvaluationItem('onset_score', f1_score=f_measure, precision_score=precision, recall_score=recall)
    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05))
    onset_offset_item = EvaluationItem('onset_offset_score', f1_score=f_measure, precision_score=precision,
                                       recall_score=recall)
    return onset_item, onset_offset_item
