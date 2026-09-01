# Object Detection — YOLOv8n on the Hailo-8 vs the i.MX8MP CPU

Runs **YOLOv8n** on the Hailo-8 (Hailo Model Zoo, compiled with HailoRT's
on-chip NMS postprocessing built into the HEF) and, as a comparison, the same
uncompiled YOLOv8n graph on the i.MX8MP Cortex-A53 cores via OpenCV's DNN
module, and draws the resulting bounding boxes.

| Command                       | Input                          | Output                                          |
|--------------------------------|---------------------------------|--------------------------------------------------|
| `object-detection-image`      | a single image (file or bundled sample) | an annotated copy, saved once, plus a text summary on stdout |
| `object-detection-camera`     | a live camera or a video file  | an annotated JPEG frame, overwritten every frame, plus a running FPS/detections line on stdout |
| `object-detection-benchmark`  | a single image (file or bundled sample) | timed latency/FPS for both backends, side by side, on stdout |

All three take `-b/--backend {hailo,cpu}` (default `hailo`; `--backend` is
repeatable on `object-detection-benchmark`, default both; `object-detection-camera`
also takes `compare`, running every backend on the same input side by side).

`object-detection-camera` also has six no-flags-needed convenience wrappers,
one per backend/source combination -- these are what you want for a quick demo:

| Command                            | Backend   | Source                    |
|--------------------------------------|-----------|---------------------------|
| `object-detection-hailo-cam`         | Hailo-8   | camera (default index 0)  |
| `object-detection-hailo-video`       | Hailo-8   | video file (1st argument) |
| `object-detection-cpu-cam`           | CPU       | camera (default index 0)  |
| `object-detection-cpu-video`         | CPU       | video file (1st argument) |
| `object-detection-compare-cam`       | both      | camera (default index 0)  |
| `object-detection-compare-video`     | both      | video file (1st argument) |

Neither `object-detection-image` nor `object-detection-camera` (nor its six
wrappers) opens a GUI window (no `cv2.imshow`): this image does not guarantee
OpenCV was built with GTK/X11 support, so instead they keep overwriting one
JPEG file that can be viewed without a local display (`scp`/`sshfs` it off the
board, or serve the directory it lives in). In `compare` mode that JPEG is the
two backends' annotated frames side by side, each labelled with its own FPS.

## Running it

```sh
object-detection-image                          # bundled sample image, Hailo-8
object-detection-image -i photo.jpg              # your own image
object-detection-image -i photo.jpg -o out.jpg   # explicit output path
object-detection-image -b cpu                    # same image, i.MX8MP CPU instead

object-detection-camera                          # camera 0, Ctrl+C to stop
object-detection-camera 1                        # a different camera index
object-detection-camera clip.mp4                  # a video file instead of a camera
object-detection-camera -o /tmp/frame.jpg         # where the annotated frame is written
object-detection-camera -n 100                    # stop after 100 frames
object-detection-camera -b cpu                    # i.MX8MP CPU instead of Hailo-8
object-detection-camera -b compare                 # both backends, side by side

# no-flags-needed wrappers around the same object-detection-camera code:
object-detection-hailo-cam                        # == object-detection-camera -b hailo
object-detection-cpu-video clip.mp4                # == object-detection-camera -b cpu clip.mp4
object-detection-compare-cam                      # == object-detection-camera -b compare

object-detection-benchmark                        # both backends, bundled sample
object-detection-benchmark -n 50                   # 50 timed inferences/backend
object-detection-benchmark -b cpu                  # just the CPU backend
```

Sample `object-detection-benchmark` output:

```
$ object-detection-benchmark
image: dog_bicycle.jpg (640x480), 20 inferences/backend

backend       load  latency       fps  detections
Hailo-8      0.42s     14.3ms     69.93  4 (dog, bicycle, truck, car)
i.MX8MP CPU  0.18s    787.0ms      1.27  3 (dog, bicycle, truck)

Hailo-8 is 55.0x faster than the CPU here.
```

(Illustrative -- exact numbers depend on the board and the HailoRT/model-zoo
version the HEF was compiled with.)

All three commands also accept `-s/--score-thres` (default `0.25`) to change
the minimum confidence drawn/reported, and `-d/--data-dir` to point at a
different model/asset directory than the installed default.

Sample output (`object-detection-image`, bundled sample):

```
$ object-detection-image
4 detection(s) in dog_bicycle.jpg:
  dog             91.2%  box=(133, 219, 314, 539)
  bicycle         88.7%  box=(103, 138, 568, 429)
  car             68.4%  box=(462, 78, 690, 172)
  person           0.0%  box=(0, 0, 0, 0)
saved annotated image to /usr/share/demo-object-detection/assets/dog_bicycle_detected.jpg
```

(Exact boxes/scores depend on the HailoRT/model-zoo version the HEF was
compiled with; treat the numbers above as illustrative.)

## Models and assets

| File              | Source                                                              |
|-------------------|----------------------------------------------------------------------|
| `hailo/yolov8n.hef` | Hailo Model Zoo v2.17.0, compiled for Hailo-8                      |
| `cpu/yolov8n.onnx`  | the same Model Zoo release's uncompiled ONNX graph (pre-NMS, Ultralytics YOLOv8n) |
| `assets/labels.txt` | COCO-80 class names, in Hailo's `Hailo-Application-Code-Examples` order |
| `assets/dog_bicycle.jpg` | Hailo's own "dog and bicycle" reference demo image             |

Everything lives under `/usr/share/demo-object-detection`; override the
location with `--data-dir`.

## Hailo-8 vs CPU: what's actually being compared

Both backends run YOLOv8n over the same 640x640 letterboxed input, tuned to
the same NMS thresholds (score 0.2, IoU 0.7 -- the Hailo HEF's compiled-in
config), so their detections on the same image line up. What differs is where
NMS happens:

- **Hailo-8** (`hailo_infer.py`): the HEF has HailoRT's on-chip NMS
  postprocessing layer built in, so the single output tensor is already a
  per-class list of decoded boxes (`postprocess.extract_detections`).
- **i.MX8MP CPU** (`cpu_infer.py`): the uncompiled ONNX graph has no
  postprocessing -- its raw `[1, 84, 8400]` output (4 box coordinates plus 80
  class scores per anchor) is decoded and NMS'd on the host, via
  `cv2.dnn.NMSBoxes` (`postprocess.extract_detections_cpu`).

Unlike [whisper-benchmark](../whisper-benchmark/README.md), there is a CPU
backend here but deliberately no i.MX8MP NPU one: YOLOv8n is exactly the kind
of quantized CNN the VeriSilicon VIP8000 NPU is built for (unlike Whisper's
transformer ops), but wiring up NXP's VX delegate/eIQ toolchain for it is a
separate effort from this comparison; `object-detection-benchmark` is
structured so an `npu` backend could be added the same way `cpu` was, if that
effort happens later.

## Why no tracking

Hailo's own
[object_detection example](https://github.com/hailo-ai/Hailo-Application-Code-Examples/blob/main/runtime/python/object_detection/README.md)
also supports `--track` (ByteTrack) across video/camera frames. That pulls in
`scipy`, `lap` and `cython_bbox` — none of which have Yocto recipes in this
BSP, and the last two are native C extensions that would need their own
cross-compile recipes. This demo intentionally stays detection-only, matching
[whisper-benchmark](../whisper-benchmark/README.md)'s approach of keeping
dependencies to what the BSP already builds (`opencv`, `numpy`,
`hailo8-python-wheels`).

## Why no on-device model auto-download

Hailo's example resolves `-n <model name>` by downloading the HEF from the
Model Zoo S3 bucket at runtime if it isn't already local. This demo instead
bundles one fixed model (`yolov8n.hef`) via `demo-object-detection-data`, the
same way `whisper-benchmark` bundles its HEFs: an embedded target should not
need network access just to run its own demo image, and pinning the exact
HEF at build time is what SRC_URI checksums are for.
