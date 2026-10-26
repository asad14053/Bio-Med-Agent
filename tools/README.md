# `tools/`: encoders and media utilities

| File | Contents | Paper |
|---|---|---|
| `encoder_imagebind.py` | `ImageBindConfig`, `ImageBindEncoder.encode_image()`, `encode_video()` | Visual encoder **f_θ**: ImageBind-huge, 224×224, CLIP normalization, L2-normalized, d = 1024 |
| `keyframe.py` | `extract_mid_frame()`, `extract_uniform_frames()` | Video preprocessing. Not used for MIMIC-CXR images, where the keyframe is the image itself. |
