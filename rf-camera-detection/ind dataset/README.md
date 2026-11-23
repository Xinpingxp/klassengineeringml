# Independent camera activity experiment

This experiment uses the staged `camera_v3ctrl_streaming` and
`camera_v3ctrl_idle` sessions to test **streaming vs. idle for one known camera**.
It does not train a camera-vs-non-camera detector.

The model filters `all_wifi_packets.csv` by the ground-truth camera MAC
`60:15:6f:2e:63:91`, rather than trusting the session `camera_mac` or
`camera_rows` fields. Session labels come from `session.json` `camera_state`;
motion sessions are excluded because their actual movement timing was not
verified. Only the v3-control sessions are staged because they share the
control-frame feature support needed for the experiment.

Features are aggregated into five-second windows over each full capture:

- BlockAck rate and camera uplink share
- Decodable camera payload rate, byte rate, size distribution, and uplink share
- Payload inter-arrival statistics

Evaluation uses leave-one-session-out cross-validation, so windows from the
held-out session never appear in that fold's training set. A final model is also
fit on all staged windows and saved for demonstration; its metrics are the
cross-validated results, not its training performance.

## Run

From the repository root:

```bash
./.venv/bin/python 'ind dataset/train_activity_model.py'
```

All outputs are written to `ind dataset/results/`.

## Important limitations

- All positive examples are from one camera, on one AP, channel, and network.
- The experiment can test this camera's recorded streaming/idle states only; it
  cannot demonstrate generalization to other cameras.
- The staged data does not include a curated set of negative devices with
  decodable payload. Payload features therefore do not establish camera-vs-
  non-camera separability.
- Idle and motion-triggered sessions are not evidence that an inactive camera
  can be detected. The honest deployment wording remains “no actively
  transmitting camera detected.”
- Session-held-out evaluation is preferable to a random window split, but 12
  sessions are still a small sample. Treat results as a pilot, not a validated
  performance estimate.